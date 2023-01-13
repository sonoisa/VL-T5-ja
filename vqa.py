# 学習済みモデルとトークナイザーを読み込む

from VLT5.vlt5_model import VLT5Model
from VLT5.vlt5_tokenizer import VLT5Tokenizer
import re
import torch


class Vqa:
    def __init__(
        self, normalize_boxes, roi_features, model_path="sonoisa/vl-t5-base-japanese"
    ):
        self.roi_features = roi_features
        self.normalized_boxes = normalize_boxes
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.vlt5 = VLT5Model.from_pretrained(model_path)
        self.vlt5.to(self.device)

        self.tokenizer = VLT5Tokenizer.from_pretrained(
            model_path, max_length=24, do_lower_case=True
        )
        self.vlt5.resize_token_embeddings(self.tokenizer.vocab_size)
        self.vlt5.tokenizer = self.tokenizer

    def get_answer(self, questions):
        self.vlt5.eval()
        box_ids = set()
        answer_list = []

        for question in questions:
            input_ids = self.tokenizer(
                question, return_tensors="pt", padding=True
            ).input_ids.to(self.device)
            vis_feats = self.roi_features.to(self.device)
            boxes = self.normalized_boxes.to(self.device)

            # 回答生成
            output = self.vlt5.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, boxes),
            )
            # 回答のトークン列を文字列に変換し、回答文に現れた画像領域IDをbox_ids変数に格納
            generated_sent = self.tokenizer.batch_decode(
                output, skip_special_tokens=False
            )[0]
            generated_sent = re.sub("[ ]*(<pad>|</s>)[ ]*", "", generated_sent)

            if "<vis_extra_id_" in generated_sent:
                match = re.match(r"<vis_extra_id_(\d+)>", generated_sent)
                box_id = int(match.group(1))
                box_ids.add(box_id)

            # ToDo: 原形に直す&ストップワード削除
            answer_list.append(generated_sent)

        return answer_list
