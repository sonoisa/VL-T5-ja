# 必要なものをインポートします
import torch
import math
import random
import logging
import warnings
from collections import defaultdict
from transformers import (
    T5Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    AutoModelForCausalLM,
    GPT2DoubleHeadsModel,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from itertools import chain
import os
from tqdm.auto import tqdm, trange
from simpletransformers.config.model_args import ConvAIArgs
from simpletransformers.conv_ai.conv_ai_utils import get_dataset
from simpletransformers.config.utils import sweep_config_to_sweep_values
import wandb
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)
SPECIAL_TOKENS = ["<s>", "</s>", "<speaker1>", "<speaker2>", "[PAD]"]
ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<PAD>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>"],
}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "labels", "token_type_ids"]


class ConvAIModelJa:
    def __init__(self, model_name, args=None, **kwargs):
        # NOTE: model_typeを削除している
        # オリジナルではMODEL_CLASSESからクラスを指定するために使用
        self.args = self._load_model_args(model_name)
        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False
        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ConvAIArgs):
            self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 日本語版のモデルに合わせる
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.tokenizer.do_lower_case = True
        # TODO: AutoModelForCausalLMにするべき？
        self.model = GPT2DoubleHeadsModel.from_pretrained(model_name)
        self.config = GPT2Config.from_pretrained(model_name)
        self.add_special_tokens_(self.model, self.tokenizer)

    def train_model(
        self,
        train_file=None,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_file=None,
        verbose=True,
        **kwargs,
    ):
        if self.args.evaluate_during_training and eval_file is None:
            warnings.warn(
                "eval_file not specified but evaluate_during_training is True. Using personachat eval data."
            )
        if args:
            self.args.update_from_dict(args)
        if not output_dir:
            output_dir = self.args.output_dir
        if (
            os.path.exists(output_dir)
            and os.listdir(output_dir)
            and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set overwrite_output_dir: True to automatically overwrite.".format(
                    output_dir
                )
            )
        self._move_model_to_device()

        train_dataloader, train_sampler = self.load_and_cache_examples(
            dataset_path=train_file,
            verbose=verbose,
            no_cache=self.args.no_cache or self.args.reprocess_input_data,
        )
        eval_loader = None
        os.makedirs(output_dir, exist_ok=True)
        global_step, training_details = self.train(
            train_dataloader,
            output_dir,
            show_running_loss=show_running_loss,
            verbose=verbose,
            **kwargs,
        )
        self.save_model(model=self.model)
        if verbose:
            logger.info(
                " Training of {} model complete. Saved to {}.".format(
                    self.args.model_type, output_dir
                )
            )

    def train(
        self,
        train_dataloader,
        output_dir,
        show_running_loss=True,
        verbose=True,
        **kwargs,
    ):
        device = self.device
        model = self.model
        args = self.args
        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)
        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd
            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)
        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )
        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = (
            warmup_steps if args.warmup_steps == 0 else args.warmup_steps
        )
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total,
        )
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            int(args.num_train_epochs), desc="Epoch", disable=args.silent
        )
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        if args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for _ in train_iterator:
            model.train()
            train_iterator.set_description(
                f"Epoch {epoch_number} of {args.num_train_epochs}"
            )
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number + 1} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                batch = tuple(t.to(device) for t in batch)
                input_ids, mc_token_ids, labels, mc_labels, token_type_ids = batch
                if args.fp16:
                    with amp.autocast():
                        outputs = model(
                            input_ids,
                            token_type_ids=token_type_ids,
                            mc_token_ids=mc_token_ids,
                            mc_labels=mc_labels,
                            labels=labels,
                        )
                        lm_loss, mc_loss = outputs[:2]
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss = lm_loss * args.lm_coef + mc_loss * args.mc_coef
                else:
                    outputs = model(
                        input_ids,
                        token_type_ids=token_type_ids,
                        mc_token_ids=mc_token_ids,
                        mc_labels=mc_labels,
                        labels=labels,
                    )
                    lm_loss, mc_loss = outputs[:2]
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    loss = lm_loss * args.lm_coef + mc_loss * args.mc_coef

                if args.n_gpu > 1:
                    loss = loss.mean()
                current_loss = loss.item()
                if show_running_loss:
                    print("\rRunning loss: %f" % current_loss, end="")
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )
                    if args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar(
                            "lr", scheduler.get_last_lr()[0], global_step
                        )
                        tb_writer.add_scalar(
                            "loss",
                            (tr_loss - logging_loss) / args.logging_steps,
                            global_step,
                        )
                        logging_loss = tr_loss
                        if args.wandb_project or self.is_sweeping:
                            wandb.log(
                                {
                                    "Training loss": current_loss,
                                    "lr": scheduler.get_last_lr()[0],
                                    "global_step": global_step,
                                }
                            )
                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )
                        self.save_model(output_dir_current, model=model)
            epoch_number += 1
            output_dir_current = os.path.join(
                output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number)
            )
            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)
            if args.save_model_every_epoch:
                self.save_model(output_dir_current, model=model)
        return (
            global_step,
            tr_loss / global_step
            if not self.args.evaluate_during_training
            else training_progress_scores,
        )

    def load_and_cache_examples(
        self,
        dataset_path=None,
        evaluate=False,
        no_cache=False,
        verbose=True,
        silent=False,
    ):
        process_count = self.args.process_count
        tokenizer = self.tokenizer
        args = self.args
        if not no_cache:
            no_cache = args.no_cache
        os.makedirs(self.args.cache_dir, exist_ok=True)
        dataset_path = dataset_path if dataset_path else ""
        dataset = get_dataset(
            tokenizer,
            dataset_path,
            args.cache_dir,
            process_count=process_count,
            proxies=self.__dict__.get("proxies", None),
            evaluate=evaluate,
            no_cache=no_cache,
            args=args,
        )
        datasets = defaultdict(list)
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and not evaluate:
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2 * args.max_history + 1) :]
                    for j, candidate in enumerate(
                        utterance["candidates"][-num_candidates:]
                    ):
                        labels = bool(j == num_candidates - 1)
                        instance = self.build_input_from_segments(
                            persona, history, candidate, tokenizer, labels
                        )
                        for input_name, input_array in instance.items():
                            datasets[input_name].append(input_array)
                    datasets["mc_labels"].append(num_candidates - 1)
                    datasets["n_candidates"] = num_candidates
                persona = [persona[-1]] + persona[:-1]  # permuted personalities

        tensor_datasets = []
        dataset = self.pad_dataset(
            datasets, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
        )
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets["n_candidates"]) + tensor.shape[1:])
            tensor_datasets.append(tensor)

        tensor_dataset = TensorDataset(*tensor_datasets)
        if not evaluate:
            data_sampler = RandomSampler(tensor_dataset)
            data_loader = DataLoader(
                tensor_dataset, sampler=data_sampler, batch_size=args.train_batch_size
            )
        else:
            data_sampler = SequentialSampler(tensor_dataset)
            data_loader = DataLoader(
                tensor_dataset, sampler=data_sampler, batch_size=args.eval_batch_size
            )
        return data_loader, data_sampler

    def compute_metrics(self, mc_preds, mc_labels, lm_logits, labels, **kwargs):
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(mc_labels, mc_preds)
        f1_current = f1_score(mc_labels.cpu().numpy(), mc_preds, average="macro")
        lm_loss_current = loss_fct(lm_logits, labels)
        return {
            **{"f1_score": f1_current, "language_model_loss": lm_loss_current},
            **extra_metrics,
        }

    def interact(self, personality=None):
        model = self.model
        args = self.args
        tokenizer = self.tokenizer
        process_count = self.args.process_count
        if self.args.fp16:
            from torch.cuda import amp
        self._move_model_to_device()
        if not personality:
            dataset = get_dataset(
                tokenizer,
                None,
                args.cache_dir,
                process_count=process_count,
                proxies=self.__dict__.get("proxies", None),
                interact=True,
                args=args,
            )
            personalities = [
                dialog["personality"]
                for dataset in dataset.values()
                for dialog in dataset
            ]
            personality = random.choice(personalities)
        else:
            personality = [tokenizer.encode(s.lower()) for s in personality]

        history = []
        while True:
            raw_text = input(">>> ")
            while not raw_text:
                print("Prompt should not be empty!")
                raw_text = input(">>> ")
            history.append(tokenizer.encode(raw_text))
            with torch.no_grad():
                if args.fp16:
                    with amp.autocast():
                        out_ids = self.sample_sequence(
                            personality, history, tokenizer, model, args
                        )
                else:
                    out_ids = self.sample_sequence(
                        personality, history, tokenizer, model, args
                    )
            history.append(out_ids)
            history = history[-(2 * args.max_history + 1) :]
            out_text = tokenizer.decode(
                out_ids, skip_special_tokens=self.args.skip_special_tokens
            )
            print("you->", raw_text)
            print("bot->", out_text)
            print("--------------------------------")
            # print(history)

    def interact_single(self, message, history, personality=None, encode_history=True):
        model = self.model
        args = self.args
        tokenizer = self.tokenizer
        process_count = self.args.process_count
        if self.args.fp16:
            from torch.cuda import amp
        self._move_model_to_device()
        if not personality:
            dataset = get_dataset(
                tokenizer,
                None,
                args.cache_dir,
                process_count=process_count,
                proxies=self.__dict__.get("proxies", None),
                interact=True,
            )
            personalities = [
                dialog["personality"]
                for dataset in dataset.values()
                for dialog in dataset
            ]
            personality = random.choice(personalities)
        else:
            personality = [tokenizer.encode(s.lower()) for s in personality]
        if encode_history:
            raw_history = history.copy()
            raw_history.append(message)
            history = [tokenizer.encode(sentence) for sentence in history]
        history.append(tokenizer.encode(message))
        with torch.no_grad():
            if args.fp16:
                with amp.autocast():
                    out_ids = self.sample_sequence(
                        personality, history, tokenizer, model, args
                    )
            else:
                out_ids = self.sample_sequence(
                    personality, history, tokenizer, model, args
                )
        out_text = tokenizer.decode(
            out_ids, skip_special_tokens=self.args.skip_special_tokens
        )
        if encode_history:
            raw_history.append(out_text)
            history = raw_history
        else:
            history.append(out_ids)
        return out_text, history

    def _threshold(self, x, threshold):
        if x >= threshold:
            return 1
        return 0

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "language_model_loss": [],
            "f1_score": [],
            **extra_metrics,
        }
        return training_progress_scores

    def save_model(self, output_dir=None, model=None, results=None):
        if not output_dir:
            output_dir = self.args.output_dir
        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            self.save_model_args(output_dir)
        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def add_special_tokens_(self, model, tokenizer):
        orig_num_tokens = 32000
        num_added_tokens = tokenizer.add_special_tokens(
            ATTR_TO_SPECIAL_TOKEN
        )  # doesn't add if they are already there
        if num_added_tokens > 0:
            self.model.resize_token_embeddings(
                new_num_tokens=orig_num_tokens + num_added_tokens
            )

    def build_input_from_segments(
        self, persona, history, reply, tokenizer, labels=False, with_eos=True
    ):
        bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS[:-1]
        )
        sequence = (
            [[bos] + list(chain(*persona))]
            + history
            + [reply + ([eos] if with_eos else [])]
        )
        sequence = [sequence[0]] + [
            [speaker2 if (len(sequence) - i) % 2 else speaker1] + s
            for i, s in enumerate(sequence[1:])
        ]
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [
            speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s
        ]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        instance["labels"] = [-100] * len(instance["input_ids"])
        if labels:
            instance["labels"] = (
                ([-100] * sum(len(s) for s in sequence[:-1]))
                + [-100]
                + sequence[-1][1:]
            )
        return instance

    def pad_dataset(self, dataset, padding=0):
        max_l = max(len(x) for x in dataset["input_ids"])
        for name in PADDED_INPUTS:
            dataset[name] = [
                x + [padding if name != "labels" else -100] * (max_l - len(x))
                for x in dataset[name]
            ]
        return dataset

    def top_filtering(
        self,
        logits,
        top_k=0.0,
        top_p=0.9,
        threshold=-float("Inf"),
        filter_value=-float("Inf"),
    ):
        assert (
            logits.dim() == 1
        )  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            # Remove all tokens with a probability less than the last token in the top-k tokens
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        if top_p > 0.0:
            # Compute cumulative probabilities of sorted tokens
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probabilities = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            # Back to unsorted indices and set them to -infinity
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value
        return logits

    def sample_sequence(
        self, personality, history, tokenizer, model, args, current_output=None
    ):
        special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        if current_output is None:
            current_output = []
        for i in range(args.max_length):
            instance = self.build_input_from_segments(
                personality, history, current_output, tokenizer, with_eos=False
            )
            input_ids = torch.tensor(
                instance["input_ids"], device=self.device
            ).unsqueeze(0)
            token_type_ids = torch.tensor(
                instance["token_type_ids"], device=self.device
            ).unsqueeze(0)
            logits = model(input_ids, token_type_ids=token_type_ids)
            logits = logits[0]
            logits = logits[0, -1, :] / args.temperature
            logits = self.top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
            probs = F.softmax(logits, dim=-1)
            prev = (
                torch.topk(probs, 1)[1]
                if not args.do_sample
                else torch.multinomial(probs, 1)
            )
            if i < args.min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    if probs.max().item() == 1:
                        warnings.warn(
                            "Warning: model generating special token with probability 1."
                        )
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)
            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())
        return current_output

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = ConvAIArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
