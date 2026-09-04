# UQ-Guided Selective Regeneration for LLM QA

LLM의 **불확실성 정량화(Uncertainty Quantification, UQ)** 지표로 오답 가능성이 높은
질문만 골라내고, 그 질문에 한해 **RAG / Recitation 프롬프트로 다시 생성**해 정확도를
끌어올리는 실험 코드입니다.

- **생성 백엔드**: vLLM — Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct,
  Ministral-8B-Instruct-2410, OLMo-2-7B-Instruct
- **데이터셋**: Natural Questions, TriviaQA, WebQuestions (+ GSM8K 로더)
- **UQ 지표**: Predictive Entropy(PE), Length-normalized PE(LNPE),
  Semantic Entropy(SE / LNSE / DSE), SAR(token / sentence)
- **채점**: RougeL > 0.3 을 정답으로 이진화 → AUROC / PR-AUC

> 코드만 담고 있습니다. 모델 가중치·생성 결과·DPR 데이터는 별도로 준비해야 합니다
> ([`third_party/README.md`](third_party/README.md) 참고).

---

## 1. 핵심 아이디어

```
                    ┌──────────────────────────────┐
   질문 ──────────► │ 1. Greedy 생성 (답변 1개)     │──► 정답 채점 (RougeL)
                    └──────────────────────────────┘
                    ┌──────────────────────────────┐
   질문 ──────────► │ 2. Sampling 생성 (답변 N=5개) │──► UQ 지표 계산
                    └──────────────────────────────┘        PE / LNPE / LNSE / SAR
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │ 3. dev set 으로 임계값 결정   │   TP − FP 최대화
                    └──────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        불확실성 < 임계값                 불확실성 ≥ 임계값
        (모델이 아는 질문)                (모델이 헷갈리는 질문)
                    │                               │
            greedy 답변 그대로            4. RAG / Recitation 프롬프트로
                                             **선택적 재생성** → 재채점
```

## 2. 저장소 구조

```
.
├── uq_paths.py         모든 입출력 경로의 단일 소스 (환경변수로 override)
├── requirements.txt    실제로 import 되는 패키지만 추린 목록
├── environment.yaml    원 실험 conda 환경 스냅샷 (Python 3.12 / CUDA 12.1)
├── .env.example        토큰 · 경로 환경변수 템플릿
│
├── vllm_pipeline/      파이프라인 전체 (생성 · 평가 · 재생성 · 분석)
│   ├── models.py       모델 레지스트리 (repo · temperature · 토큰 패턴)
│   ├── common.py       슬라이싱 · rouge 공용 헬퍼
│   ├── psg_generator.py  DPR 패시지에서 질문별 top-5 추출 (rag 포맷 준비용)
│   └── *.sh            모델 × 데이터셋 × 포맷 실행 조합
├── docs/
│   ├── PIPELINE.md     실행 순서 · 스크립트별 입출력 상세
│   └── STRUCTURE.md    파일 단위 설명
├── third_party/        DPR 등 외부 저장소 · 데이터 설치 안내
└── results/            산출물 저장 위치 (git 추적 제외)
```

모든 실행 스크립트는 `vllm_pipeline/` 안에 있습니다.

## 3. 설치

```bash
conda env create -f environment.yaml     # 원 환경 그대로 재현 (권장)
conda activate vllm_inference
# 또는
pip install -r requirements.txt
```

### 환경변수

```bash
cp .env.example .env
# .env 편집 후
set -a && . ./.env && set +a
```

| 변수 | 용도 | 기본값 |
|---|---|---|
| `HF_TOKEN` | gated 모델(Llama 등) 다운로드 | — (필수) |
| `UQ_RESULTS_DIR` | 산출물 루트 | `./results/vllm` |
| `DPR_DATA_DIR` | DPR 검색 결과 · 위키 패시지 | `./third_party/DPR/dpr/data/downloads/data` |

```bash
python uq_paths.py                                  # 해석된 경로 확인
python -c "import uq_paths; uq_paths.ensure_dirs()" # 산출물 디렉터리 생성
```

`nq` · `webq` 데이터와 `rag` 포맷 문맥은 DPR 검색 결과 파일이 필요합니다.
[`third_party/README.md`](third_party/README.md) 를 먼저 따라 주세요.

## 4. 빠른 실행

스크립트는 **자기 디렉터리 안에서** 실행합니다 (`import data_loader_vllm` 등
같은 폴더 모듈에 의존).

```bash
cd vllm_pipeline

# 1) 생성: greedy 1개 + sampling 5개
python greedy_vllm.py   --model_name llama3.1-8b-it --data_name nq --format direct --num_gen 5 --device_map 0,1
python sampling_vllm.py --model_name llama3.1-8b-it --data_name nq --format direct --num_gen 5 --device_map 0,1

# 2) 후처리 + 채점 + UQ 지표 계산
python greedy_eval_vllm.py   --model_name llama3.1-8b-it --data_name nq --format direct --num_gen 5 --device_map 0,1
python sampling_eval_vllm.py --model_name llama3.1-8b-it --data_name nq --format direct --num_gen 5 --device_map 0,1

# 3) 불확실성 임계값 초과 질문만 RAG 로 재생성
python regen_fast.py --model_name llama3.1-8b-it --data_name nq \
                     --format direct --format2 rag --look vote --device_map 0,1

# 4) AUROC / PR-AUC 리포트
python curve_return_vllm.py --model_name llama3.1-8b-it --data_name nq --format direct
```

`vllm_pipeline/*.sh` 에 모델 × 데이터셋 × 포맷 조합이 정리되어 있습니다.
대부분 주석 처리되어 있으니 필요한 줄만 해제해서 쓰세요.

### 주요 인자

| 인자 | 값 |
|---|---|
| `--model_name` | `llama3.1-8b-it`, `qwen2.5-7b-it`, `mistral-8b-it`(= Ministral-8B), `olmo2-7b-it` |
| `--data_name` | `nq`, `triviaqa`, `webq`, `gsm8k` |
| `--format` | `direct`, `reflect2`, `paraphrase`, `para-under`, `rec`, `rag` |
| `--format2` | 재생성 프롬프트: `rag`, `rec`, `direct` |
| `--look` | 재생성 트리거: `pe`, `lnpe`, `lnse`, `sar`, `one`, `two`, `three`, `vote` |
| `--num_gen` | sampling 생성 개수 (기본 5) |
| `--device_map` | `CUDA_VISIBLE_DEVICES` 로 넘어가는 GPU 목록 (예: `0,1`) |
| `--split` | `test`(기본) / `dev` / `ablation` — 산출물이 놓이는 갈래 |

> `--device_map` 은 결과 파일명의 `d01` / `d23` 접두어도 결정합니다.
> 생성과 평가에서 **같은 값**을 써야 파일을 찾을 수 있습니다.

전체 실행 순서와 스크립트별 입출력은 [`docs/PIPELINE.md`](docs/PIPELINE.md) 참고.

## 5. 주의사항

- **임계값이 코드에 하드코딩되어 있습니다.** `vllm_pipeline/utils_vllm.py` 의
  `load_threshold()` 는 원 실험 dev set 에서 구한 값입니다. 다른 모델·데이터셋을 쓴다면 `threshold_vllm.py` 로 다시 구해
  이 표를 갱신해야 합니다.
- **새 모델 추가**는 `vllm_pipeline/models.py` 의 `MODELS` 에만 넣으면 됩니다.
- 결과 파일명 규칙은 [`results/README.md`](results/README.md) 를 보세요.

## 6. 참고 문헌

- Kuhn et al., *Semantic Uncertainty* (ICLR 2023) — semantic entropy
- Duan et al., *Shifting Attention to Relevance (SAR)* (ACL 2024) — [jinhaoduan/SAR](https://github.com/jinhaoduan/SAR)
- Karpukhin et al., *Dense Passage Retrieval* (EMNLP 2020) — [facebookresearch/DPR](https://github.com/facebookresearch/DPR)

## 7. 라이선스

[MIT](LICENSE)
