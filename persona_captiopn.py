import logging
import scipy.spatial
from object_detection import ObjectDetection
from sentence_bert import SentenceBertJapanese
from vqa import Vqa

logger = logging.getLogger(__name__)


class PersonaCaption:
    def __init__(self, model_path="sonoisa/sentence-bert-base-ja-mean-tokens"):
        self.model_path = model_path
        self.model = model = SentenceBertJapanese(model_path)

    def _get_word_list(self, image_path):
        d = ObjectDetection()

        output = d.detection(image_path)
        labels = d.get_object_labels(output)
        normalized_boxes, roi_features = d.get_object_features_for_vlt5(output)

        v = Vqa(normalize_boxes=normalized_boxes, roi_features=roi_features)
        questions = []
        with open("./data/questions.txt") as f:
            for question in f.readlines():
                questions.append(question)
        answers = v.get_answer(questions)

        word_list = list(set(labels + answers))
        return word_list

    def _search(self, queries, top_n=5):
        logger.info("Searching...")
        sentences = []
        labels = []
        with open("./data/persona_list.csv") as f:
            for i, text in enumerate(f):
                if i == 0:
                    continue
                print(i, "...", end="\r")
                persona = text.split(",")
                desc = persona[1].strip()
                label = persona[2].strip()
                sentences.append(desc)
                labels.append(label)
        sentence_vectors = self.model.encode(sentences)
        query_embeddings = self.model.encode(queries).numpy()

        search_result = []
        for query, query_embedding in zip(queries, query_embeddings):
            distances = scipy.spatial.distance.cdist(
                [query_embedding], sentence_vectors, metric="cosine"
            )[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            # print("\n\n======================\n\n")
            # print("Query:", query)
            # print("\nTop 5 most similar sentences in corpus:")

            for idx, distance in results[0:top_n]:
                # print(
                #     labels[idx],
                #     sentences[idx].strip(),
                #     "(Distance: %.4f)" % (1 / distance),
                # )
                search_result.append(sentences[idx].strip())

        return search_result

        # TODO: 何かしらの検索結果を返す

    def get_caption(self, image_path):
        word_list = self._get_word_list(image_path)
        search_result = self._search(word_list, 1)
        logger.info("Successfully get persona caption")
        return search_result
        # TODO: 何かしらの基準でペルソナを選ぶ
