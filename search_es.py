import sys
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")


def search(desc):
    results = es.search(index="persona_list", query={"match": {"desc": desc}})

    print("検索語: {}".format(desc))
    print("検索結果数: {}".format(len(results["hits"]["hits"])))
    print()

    for result in results["hits"]["hits"]:
        print(
            result["_source"]["desc"],
            result["_score"],
        )


if __name__ == "__main__":
    desc = sys.argv[1]
    search(desc)
