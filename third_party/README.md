# 외부 저장소 / 데이터

이 디렉터리는 비어 있습니다. 아래 항목을 여기에 직접 설치하세요.
`.gitignore` 로 추적에서 제외되어 있습니다.

## DPR — Dense Passage Retrieval (필수)

`nq` · `webq` 데이터셋 로딩과 `rag` 포맷의 문맥이 DPR 검색 결과에 의존합니다.

```bash
cd third_party
git clone https://github.com/facebookresearch/DPR.git
cd DPR && pip install .

# 검색 결과 및 위키 패시지 다운로드
python data/download_data.py --resource data.retriever_results.nq.single.test
python data/download_data.py --resource data.retriever.qas.webq
python data/download_data.py --resource data.wikipedia_split.psgs_w100
```

코드가 기대하는 경로 (`DPR_DATA_DIR` 로 변경 가능):

```
third_party/DPR/dpr/data/downloads/data/
├── retriever_results/nq/single/test.json
├── retriever/webq_mytest.pkl          # psg_generator.py 로 생성
├── retriever/triviaqa_mytest.pkl      # psg_generator.py 로 생성
└── wikipedia_split/psgs_w100.tsv
```

`*_mytest.pkl` 은 `vllm_pipeline/psg_generator.py` 로 질문별 top-5 패시지를 뽑아
만드는 파일입니다 (`faiss` 필요).

## SAR — Shifting Attention to Relevance (참고용)

SAR 지표는 `vllm_pipeline/uq_vllm.py` 에 직접 구현되어 있습니다.
원 논문 구현과 비교하려면:

```bash
cd third_party
git clone https://github.com/jinhaoduan/SAR.git
```

## 사용된 모델 (HuggingFace Hub)

가중치는 포함하지 않습니다. 최초 실행 시 자동으로 내려받습니다 (`HF_TOKEN` 필요).

```
meta-llama/Meta-Llama-3.1-8B-Instruct
Qwen/Qwen2.5-7B-Instruct
mistralai/Ministral-8B-Instruct-2410
allenai/OLMo-2-1124-7B-Instruct
microsoft/deberta-large-mnli            # 의미 클러스터링
cross-encoder/stsb-roberta-large        # SAR 토큰 중요도
```
