# 파일 단위 설명

## `vllm_pipeline/`

**공용 모듈**

| 파일 | 설명 |
|---|---|
| `models.py` | 모델 레지스트리 — repo · temperature · 정답 슬라이싱 토큰 패턴의 **단일 소스**. 새 모델은 여기만 추가한다. |
| `common.py` | `slice_start_finder` / `slice_end_finder` / `max_rouge_scores` |
| `paths.py` | 저장소 루트 `uq_paths.py` 로 위임하는 shim (`result_path`, `device_tag` 등) |
| `psg_generator.py` | DPR 위키 패시지에서 질문별 top-5 패시지 추출 — `rag` 포맷과 `webq` 로딩이 쓰는 `*_mytest.pkl` 을 만든다 (`faiss` 필요) |
| `data_loader_vllm.py` | 데이터셋 로딩 + 모든 프롬프트 템플릿 (direct / reflect2 / paraphrase / para-under / rec / rag) |
| `uq_vllm.py` | UQ 지표 구현: PE, LNPE, SE/LNSE, DSE, SAR, `ClassifyWrapper`(NLI 클러스터링), RougeL |
| `utils_vllm.py` | AUROC, 임계값 탐색(`th_vals`, `optimal_threshold_max_tp_minus_fp`), 임계값 표 `load_threshold` |

**실행 스크립트**

| 파일 | 설명 |
|---|---|
| `greedy_vllm.py` | greedy 생성. `--split {test,ablation}` |
| `sampling_vllm.py` | sampling 생성 (N개). `--split {test,ablation}` |
| `greedy_vllm_dev.py` / `sampling_vllm_dev.py` | dev split 생성 — DPR dev 파일을 직접 읽으므로 별도 유지 |
| `greedy_eval_vllm.py` | greedy 후처리 + RougeL 채점. `--split {test,dev,ablation}` |
| `sampling_eval_vllm.py` | sampling 후처리 + UQ 지표 계산. `--split {test,dev,ablation}` |
| `threshold_vllm.py` | dev set 으로 지표별 임계값 산출 (결과를 표 형태로 출력) |
| `regen_fast.py` | **선택적 재생성 (핵심)** |
| `gen_kig_vllm.py` | 전량 재생성 비교군 |
| `curve_return_vllm.py` | AUROC / PR-AUC. `--split {test,ablation}` |
| `*.sh` | 모델 × 데이터셋 × 포맷 실행 조합 (대부분 주석 처리) |
