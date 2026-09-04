# results/

파이프라인 산출물이 쌓이는 위치입니다. **git 으로 추적하지 않습니다**
(`.gitignore` 에서 `/results/*` 제외).

원 실험에서는 약 **12GB** 의 `.pkl` 이 생성되었습니다. 저장소 밖에 두려면
`UQ_RESULTS_DIR` 환경변수를 설정하세요.

```
results/
├── vllm/                        ← UQ_RESULTS_DIR (기본값)
│   ├── results/
│   │   ├── not_post_processed/  Stage 1 원본 생성 결과
│   │   ├── post_processed/      Stage 2 채점 + UQ 지표
│   │   ├── double_loop/         Stage 4 선택적 재생성
│   │   └── single_kig/          전량 재생성 비교군
│   ├── dev_results/
│   │   ├── not_post_processed/
│   │   └── post_processed/      Stage 3 임계값 산출 입력
│   └── ablation_results/
│       └── posted_results/
```

디렉터리 생성:

```bash
python -c "import uq_paths; uq_paths.ensure_dirs()"
```

## 파일명 규칙

```
{data_name}-{d01|d23}-gen{num_gen}-{model_name}-{format}.pkl
sampling-{data_name}-{d01|d23}-gen{num_gen}-{model_name}-{format}.pkl

# Stage 4
{data_name}-{d01|d23}-gen{num_gen}-{model_name}-{format}-{format2}-{look}.pkl
```

`d01` / `d23` 는 `--device_map` 에서 파생됩니다. 생성과 평가에서 같은 값을 써야 합니다.
