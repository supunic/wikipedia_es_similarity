import gzip
import json
from typing import Dict, Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from gensim.models import KeyedVectors
from joblib import Parallel, delayed

from swem import MeCabTokenizer
from swem import SWEM


def index_batch(docs):
    # 並列処理プロセスの実行
    requests = Parallel(n_jobs=1)([delayed(get_request)(doc) for doc in docs])
    bulk(client, requests)


def get_request(doc) -> Dict[str, Any]:
    return {
        "_op_type": "index",
        "_index": INDEX_NAME,
        "text": doc["text"],
        "title": doc["title"],
        "text_vector": swem.average_pooling(doc["text"]).tolist()
    }


# embedding
w2v_path = "jawiki.word_vectors.200d.txt"
w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
tokenizer = MeCabTokenizer("-O wakati")
swem = SWEM(w2v, tokenizer)

# elasticsearch
client = Elasticsearch()
BATCH_SIZE = 1000
INDEX_NAME = "wikipedia"

# indexの削除
client.indices.delete(index=INDEX_NAME, ignore=[404])

# index "wikipedia" の作成
with open("index.json") as index_file:
    source = index_file.read().strip()
    json_source = json.loads(source)
    client.indices.create(index=INDEX_NAME, settings=json_source["settings"], mappings=json_source["mappings"])

docs = []
count = 0
with gzip.open("jawiki-20211115-cirrussearch-content.json.gz") as f:
    for line in f:
        json_line = json.loads(line)

        # "index"が入っているラインは無視
        if "index" not in json_line:
            doc = json_line
            docs.append(doc)
            count += 1

            # バッチサイズまでdocsが増えたらbulk処理を実行
            if count % BATCH_SIZE == 0:
                index_batch(docs)
                docs = []
                print(f"Indexed {count} documents. {100.0 * count / 1165654}%")

    # 残りのdocsのbulk処理を実行
    if docs:
        index_batch(docs)
        print("Indexed {} documents.".format(count))
