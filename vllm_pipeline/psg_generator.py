# DPR 위키 패시지에서 질문별 top-5 를 뽑아 {data}_mytest.pkl 을 만든다.
# data_loader_vllm.py 의 webq 로딩과 rag 포맷이 이 파일을 쓴다.
#
# 데이터셋 전환은 아래 세 곳의 주석을 함께 바꿔서 한다:
#   (1) 데이터 로딩  (2) data_dict 의 'answers'  (3) save_path

import paths
import pandas as pd
path = f"{paths.DPR_DATA}/wikipedia_split/psgs_w100.tsv"
df = pd.read_csv(path, sep="\t")


# title + passage 합성
from tqdm import tqdm

text_list = []
for i in tqdm(range(len(df['text']))):
    try:
        text_list.append(df['title'][i] + ' ' + df['text'][i])
    except:
        text_list.append(df['text'][i])


# webq 데이터 로딩
# import json
# webq_dev_path = f"{paths.DPR_DATA}/retriever/webq-dev.json"
# with open(webq_dev_path, 'r') as f:
# 	webq_data_dev = json.load(f)

# val_query_list = []
# val_answer_list = []
# for i in range(len(webq_data_dev)):
#     val_query_list.append(webq_data_dev[i]['question'])
#     val_answer_list.append(webq_data_dev[i]['answers'])

# 데이터 로딩
import json
from datasets import load_dataset

# webq data loading
# webq_data_test = load_dataset("Stanford/web_questions")
# webq_data_test = webq_data_test['test'] # question, answers
# test_data = webq_data_test # question, answers

# test_query_list = webq_data_test['question']

# nq data loading
# path = f"{paths.DPR_DATA}/retriever_results/nq/single/test.json"
# with open(path, 'r') as f:
#     data = json.load(f)

# triviaqa data loading
data = load_dataset("trivia_qa", "rc.nocontext", split="validation")
id_mem = set()
def remove_dups(batch):
    if batch['question_id'][0] in id_mem:
        return {_:[] for _ in batch.keys()}
    id_mem.add(batch['question_id'][0])
    return batch
data = data.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
data = data.remove_columns(["search_results", "question_source", "entity_pages"])

test_query_list = []
for idx, d in enumerate(data):
    q = d['question'] if d['question'].endswith("?") else d['question'] + "?"
    test_query_list.append(q)


from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import time

# document 준비
documents = text_list

# 임베딩 모델 로드
model = SentenceTransformer("all-MiniLM-L6-v2")

# 문서 임베딩 및 정규화(for cosine similarity)
doc_embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
doc_embeddings = doc_embeddings/np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

# indexflatip 생성 및 문서 추가
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings) # 모든 문서 등록

# 문서 갯수 출력
print(f"FAISS Index size: {index.ntotal} documents")

# 쿼리 임베딩 및 검색(validation)
val_dataset = []
queries = test_query_list
top_k = 5
for i, query in enumerate(tqdm(queries)):
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding) # 정규화
    D, I = index.search(query_embedding, top_k)
    score_list = []
    doc_list = []
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        score_list.append(score)
        doc_list.append(documents[idx])
        
    data_dict = {'scores' : score_list,
                 'docs' : doc_list,
                 'question' : query,
                 'answers': data[i]['answer']['normalized_aliases']} # trivia 전용
                 #  'answers' : data[i]['answers']} # nq 전용
                 # 'answers' : webq_data_test['answers'][i]} # webq 전용
    
    val_dataset.append(data_dict)
    
# val 데이터 저장(pickle)
import pickle
# save_path = f"{paths.DPR_DATA}/retriever/webq_mytest.pkl"
# save_path = f"{paths.DPR_DATA}/retriever/nq_mytest.pkl"
save_path = f"{paths.DPR_DATA}/retriever/triviaqa_mytest.pkl"
with open(save_path, "wb") as json_file:
    pickle.dump(val_dataset, json_file)