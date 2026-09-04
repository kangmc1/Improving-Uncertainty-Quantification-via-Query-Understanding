# 파이프라인 상세

`vllm_pipeline/` 기준입니다. 모든 명령은 해당 디렉터리 안에서 실행합니다.

경로 기호:
- `$OUT` = `paths.VLLM_OUT` (기본 `results/vllm`, `UQ_RESULTS_DIR` 로 override)
- `$NAME` = `{data_name}-{device_path}-gen{num_gen}-{model_name}-{format}`
  - `device_path` 는 `--device_map` 에 `0`/`1` 이 있으면 `d01`, `2`/`3` 이 있으면 `d23`

---

## Stage 1 — 생성

| 스크립트 | 하는 일 | 출력 |
|---|---|---|
| `greedy_vllm.py` | 질문당 답변 1개 생성 (모델별 고정 temperature, logprob 포함) | `$OUT/results/not_post_processed/$NAME.pkl` |
| `sampling_vllm.py` | 질문당 답변 `--num_gen` 개 생성 | `$OUT/results/not_post_processed/sampling-$NAME.pkl` |
| 〃 `--split ablation` | paraphrase · para-under ablation | `$OUT/ablation_results/…` (파일명에 device tag 없음) |
| `greedy_vllm_dev.py` / `sampling_vllm_dev.py` | 임계값 산출용 dev split 생성 | `$OUT/dev_results/not_post_processed/…` |

> dev 생성만 별도 스크립트다. DPR 의 dev 파일(`nq-dev.json`, `trivia-dev.json`,
> `webq_mydev.pkl`)을 직접 읽고 문맥을 따로 구성하기 때문에 `--split` 으로 합칠 수 없다.

프롬프트 템플릿은 전부 `data_loader_vllm.py` 안에 있습니다.

| `--format` | 함수 | 설명 |
|---|---|---|
| `direct` | `original_prompting` | 질문 이해 확인 후 바로 답변 |
| `reflect2` | `reflection_prompting2` | 2-step reflection |
| `para-under` | `reflection_prompting` | 질문 재진술 기반 이해 확인 |
| `paraphrase` | `para_prompting` | 질문 패러프레이즈 |
| `rec` | `recitation_prompting` | 관련 지식을 4-step 으로 recite 후 답변 |
| `rag` | `rag_prompting` | DPR 검색 패시지를 붙여 답변 |

데이터 로더(`data_loading`)는 `trivia`(HF `trivia_qa` validation),
`nq`·`webq`(DPR 검색 결과 파일), `gsm8k`(HF `openai/gsm8k`) 를 지원합니다.

## Stage 2 — 후처리 · 채점 · UQ 계산

평가는 `--split {test,dev,ablation}` 하나로 세 갈래를 모두 처리한다.

| 스크립트 | 하는 일 | 출력 |
|---|---|---|
| `greedy_eval_vllm.py` | 모델별 토큰 패턴으로 `[Answer]` 이후만 slicing → RougeL 채점 | `$OUT/results/post_processed/$NAME.pkl` |
| `sampling_eval_vllm.py` | N개 응답의 유사도 클러스터링 → PE / LNPE / SE / LNSE / DSE / SAR 계산 | `$OUT/results/post_processed/sampling-$NAME.pkl` |
| 〃 `--split dev` | 위와 동일, dev split | `$OUT/dev_results/post_processed/…` |
| 〃 `--split ablation` | ablation split | `$OUT/ablation_results/posted_results/…` |

UQ 지표 구현은 전부 `uq_vllm.py` 에 있습니다. 의미 클러스터링은
`microsoft/deberta-large-mnli`, SAR 의 토큰 중요도는 `cross-encoder/stsb-roberta-large`
를 씁니다.

## Stage 3 — 임계값 결정

```bash
python threshold_vllm.py --model_name llama3.1-8b-it --data_name nq --format direct --num_gen 5 --device_map 2,3
```

`dev_results/post_processed` 의 greedy·sampling 결과를 병합한 뒤,
`correct = (rougel_score > 0.3)` 로 이진화하고 **TP − FP 를 최대화**하는 분위수 지점을
지표별로 찾습니다 (`utils_vllm.optimal_threshold_max_tp_minus_fp` 가 개선판).

스크립트는 지표별 임계값과 `[pe, lnpe, lnse, sar]` 순서의 리스트를 출력합니다.
이 값을 아래 표에 옮겨 적습니다.

`utils_vllm.load_threshold(data_name, model_name, generation_name)` 가
`d{데이터}{모델}{생성포맷}` 키로 `[pe, lnpe, lnse, sar]` 를 돌려줍니다.

**새 조합을 돌리면 이 표를 직접 갱신해야 합니다.**

## Stage 4 — 선택적 재생성 (핵심)

```bash
python regen_fast.py --model_name llama3.1-8b-it --data_name nq \
                     --format direct --format2 rag --look vote --device_map 0,1
```

1. `results/post_processed` 의 greedy·sampling 결과를 로드
2. `--look` 규칙으로 **재생성 대상 질문**을 선별
   - `pe` / `lnpe` / `lnse` / `sar` : 해당 지표 단독 임계값 초과
   - `one` : PE 만
   - `two` : PE·LNPE 중 1개 이상
   - `three` : PE·LNPE·LNSE 중 2개 이상
   - `vote`(기본) : PE·LNPE·LNSE·SAR 중 2개 이상
3. 선별된 질문만 `--format2`(`rag` / `rec` / `direct`) 프롬프트로 재생성
4. 재생성 답변을 RougeL 재채점하고, 나머지는 원래 점수를 유지
5. 전체 데이터에 `rougel_score_{look}` 컬럼을 추가해 저장

출력: `$OUT/results/double_loop/{data}-{dev}-gen{N}-{model}-{format}-{format2}-{look}.pkl`

비교군으로 **모든 질문을 재생성**하는 `gen_kig_vllm.py`
(출력 `$OUT/results/single_kig/…`) 가 있습니다.

## Stage 5 — 분석

`curve_return_vllm.py` 로 지표별 AUROC 와 PR-AUC 를 뽑습니다.

---

각 단계의 모델 × 데이터셋 × 포맷 조합은 `vllm_pipeline/` 의 `.sh` 파일에 정리되어
있습니다. 파일 이름이 곧 용도이며(`vllm_regen_fast.sh` → Stage 4 등), 대부분의 줄은
주석 처리되어 있으니 필요한 조합만 해제해서 쓰세요.
