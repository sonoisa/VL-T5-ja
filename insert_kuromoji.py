from elasticsearch import Elasticsearch, helpers

es = Elasticsearch("http://localhost:9200")


def load():
    if es.indices.exists(index="persona_list"):
        es.indices.delete(index="persona_list", ignore=[400, 404])

    # kuromojiで形態素解析した結果で転置インデックスを作成
    mapping = {
        "properties": {
            "desc": {"type": "text", "analyzer": "kuromoji"},
        }
    }
    es.indices.create(index="persona_list", mappings=mapping)

    with open("./data/persona_list.csv") as f:
        for i, text in enumerate(f):
            print(i, "...", end="\r")
            persona = text.split(",")
            desc = persona[1].strip()
            item = {
                "_index": "persona_list",
                "_source": {"desc": desc},
            }
            yield item


if __name__ == "__main__":
    print(helpers.bulk(es, load()))
