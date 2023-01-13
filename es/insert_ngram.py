from elasticsearch import Elasticsearch, helpers

es = Elasticsearch("http://localhost:9200")


def load():
    if es.indices.exists(index="persona_list"):
        es.indices.delete(index="persona_list", ignore=[400, 404])

    # 2gramで分割した結果で転置インデックスを作成
    # TODO: これでできてるのか？
    settings = {
        "analysis": {
            "analyzer": {
                "my_analyzer": {"type": "custom", "tokenizer": "my_tokenizer"}
            },
            "tokenizer": {
                "my_tokenizer": {
                    "type": "ngram",
                    "min_gram": 2,
                    "max_gram": 2,
                    "token_chars": ["letter", "digit"],
                }
            },
        }
    }

    mapping = {
        "properties": {
            "desc": {"type": "text", "analyzer": "my_analyzer"},
        }
    }

    es.indices.create(index="persona_list", mappings=mapping, settings=settings)

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
