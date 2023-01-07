from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

results = es.search(
            index="persona_list",
            body={
                "query": {"match": {"desc": "自転車"}},
                "size": 100,
            },
        )

for result in results["hits"]["hits"]:
  print(
    result["_source"]["desc"],
    result["_score"],
  )
