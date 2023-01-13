import scipy.spatial
from sentence_bert import SentenceBertJapanese

model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens")

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
sentence_vectors = model.encode(sentences)


def search(queries):
    query_embeddings = model.encode(queries).numpy()

    closest_n = 5
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist(
            [query_embedding], sentence_vectors, metric="cosine"
        )[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for idx, distance in results[0:closest_n]:
            print(labels[idx], sentences[idx].strip(), "(Distance: %.4f)" % (distance))


queries = ["雪", "車", "男性", "老人", "犬", "大阪"]
search(queries)
