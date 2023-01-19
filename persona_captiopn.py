import logging
import scipy.spatial
from object_detection import ObjectDetection
from sentence_bert import SentenceBertJapanese
from vqa import Vqa
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)


class PersonaCaption:
    def __init__(self, model_path="sonoisa/sentence-bert-base-ja-mean-tokens"):
        self.model_path = model_path
        self.model = SentenceBertJapanese(model_path)

        self.persona_data = {}
        with open("./data/persona_list.csv") as f:
            for i, text in enumerate(f):
                if i == 0:
                    continue
                print(i, "...", end="\r")
                persona = text.split(",")
                desc = persona[1].strip()
                label = persona[2].strip()
                self.persona_data[desc] = label

    def _get_word_list(self, image_path):
        object_detection = ObjectDetection()

        output = object_detection.detection(image_path)
        labels = object_detection.get_object_labels(output)
        normalized_boxes, roi_features = object_detection.get_object_features_for_vlt5(
            output
        )

        vqa = Vqa(normalize_boxes=normalized_boxes, roi_features=roi_features)
        questions = []
        with open("./data/questions.txt") as f:
            for question in f.readlines():
                questions.append(question)
        answers = list(set(vqa.get_answer(questions)))

        word_list = list(set(labels + answers))
        logger.info("Successfully build word list. word list = %s", word_list)
        return word_list

    def _get_word_score_dict(self, image_path, output_size=5):
        word_list = self._get_word_list(image_path)
        # 物体検出とVQAで得た単語のスコアは1とする
        word_score_dict = {w: 1.0 for w in word_list}
        word2vec_model = KeyedVectors.load(
            "./data/chive-1.2-mc30_gensim/chive-1.2-mc30.kv"
        )
        for query in list(word_score_dict.keys()):
            if query in word2vec_model:
                for sim in word2vec_model.most_similar(query, topn=output_size):
                    word = sim[0]
                    score = round(float(sim[1]), 3)
                    if word not in word_score_dict or (
                        word in word_score_dict and score > word_score_dict[word]
                    ):
                        word_score_dict[word] = score
        logger.info("Successfully build word score dict. dict = %s", word_score_dict)
        return word_score_dict

    def _search(self, word_score_dict, distance_threshold=1):
        # ペルソナ文を検索し、結果のペルソナ文にスコアを設定して返す
        logger.info("Searching...")
        search_queries = list(word_score_dict.keys())
        persona_sentences = list(self.persona_data.keys())
        sentence_vectors = self.model.encode(persona_sentences)
        query_embeddings = self.model.encode(search_queries).numpy()

        search_result = {}
        for query, query_embedding in zip(search_queries, query_embeddings):
            query_score = word_score_dict[query]
            distances = scipy.spatial.distance.cdist(
                [query_embedding], sentence_vectors, metric="cosine"
            )[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])
            for idx, distance in results:
                ## TODO:全文についてスコア計算をしても良い
                if distance < distance_threshold:
                    persona_sentence = persona_sentences[idx]
                    persona_score = self._get_persona_score(query_score, distance)
                    if persona_sentence not in search_result or (
                        persona_sentence in search_result
                        and persona_score > search_result[persona_sentence]
                    ):
                        search_result[persona_sentence] = persona_score
        search_result = sorted(search_result.items(), key=lambda x: x[1], reverse=True)
        logger.info("Successfully search by queries. result = %s", search_result)
        return search_result

    def _get_persona_score(self, word_score, distance):
        # TODO: もう少しword_scoreの変化を大きく変更できないか？
        return word_score / (distance + 1)

    def get_caption(self, image_path, persona_num):
        # TODO: カテゴリの重複削除→含意関係認識
        word_score_dict = self._get_word_score_dict(image_path)
        search_result = self._search(word_score_dict)
        persona_list = []
        label_result = []
        for result in search_result:
            label = self.persona_data[result[0]]
            # カテゴリの重複削除
            if label not in label_result:
                persona_list.append(result[0])
                label_result.append(label)
                if len(persona_list) >= persona_num:
                    break

        logger.info(
            "Successfully get persona caption. persona caption = %s", persona_list
        )
        return persona_list
