"""저장소 전역 경로 설정 (단일 소스).

모든 경로는 환경변수로 덮어쓸 수 있다. 아무것도 지정하지 않으면
저장소 루트 기준의 기본값을 사용한다.

    export UQ_RESULTS_DIR=/mnt/big/uq_results   # 파이프라인 산출물 (대용량)
    export DPR_DATA_DIR=/mnt/big/DPR/dpr/data/downloads/data
    export UQ_REPO_ROOT=/path/to/this/repo      # 보통 자동 인식되므로 불필요

산출물은 수 GB 단위로 쌓이므로, 저장소 밖 경로를 가리키게 하는 것을 권장한다.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

REPO_ROOT = os.environ.get("UQ_REPO_ROOT", _HERE)

# 파이프라인 산출물 루트.
# 하위에 results/{not_post_processed,post_processed,double_loop,single_kig},
# dev_results/{...}, ablation_results/ 가 생성된다.
VLLM_OUT = os.environ.get("UQ_RESULTS_DIR", os.path.join(REPO_ROOT, "results", "vllm"))


# DPR 검색 결과 / 위키 패시지 (third_party/README.md 참고).
DPR_DATA = os.environ.get(
    "DPR_DATA_DIR",
    os.path.join(REPO_ROOT, "third_party", "DPR", "dpr", "data", "downloads", "data"),
)


# --------------------------------------------------------------- 결과 경로 규약
# split 마다 산출물이 놓이는 디렉터리와 파일명 규칙이 다르다.
#   test     : results/{not_post_processed,post_processed}/   파일명에 device tag 포함
#   dev      : dev_results/{not_post_processed,post_processed}/   device tag 포함
#   ablation : ablation_results/ 와 ablation_results/posted_results/   device tag 없음
_SPLIT_LAYOUT = {
    "test":     {"root": "results",          "raw": "not_post_processed", "scored": "post_processed", "tagged": True},
    "dev":      {"root": "dev_results",      "raw": "not_post_processed", "scored": "post_processed", "tagged": True},
    "ablation": {"root": "ablation_results", "raw": None,                 "scored": "posted_results", "tagged": False},
}

SPLITS = tuple(_SPLIT_LAYOUT)
STAGES = ("raw", "scored")


def device_tag(device_map):
    """--device_map 문자열에서 결과 파일명에 쓰이는 태그를 만든다.

    기존 규칙을 그대로 유지한다: 0 또는 1 이 있으면 d01, 그 다음 2 또는 3 이면 d23.
    ("0,1,2,3" 이 d01 이 되는 것도 기존 동작 그대로다.)
    """
    if "0" in device_map or "1" in device_map:
        return "d01"
    if "2" in device_map or "3" in device_map:
        return "d23"
    raise ValueError(
        f"--device_map '{device_map}' 에서 결과 파일 태그를 정할 수 없다. "
        "0~3 중 하나 이상을 포함해야 한다."
    )


def _layout(split):
    try:
        return _SPLIT_LAYOUT[split]
    except KeyError:
        raise KeyError(f"알 수 없는 --split '{split}'. 사용 가능: {', '.join(SPLITS)}") from None


def stage_dir(split, stage):
    """산출물 디렉터리. stage 는 'raw'(생성 직후) 또는 'scored'(채점·UQ 계산 후)."""
    lay = _layout(split)
    if stage not in STAGES:
        raise KeyError(f"알 수 없는 stage '{stage}'. 사용 가능: {', '.join(STAGES)}")
    sub = lay[stage]
    parts = [VLLM_OUT, lay["root"]] + ([sub] if sub else [])
    return os.path.join(*parts)


def result_name(data_name, model_name, fmt, num_gen, tag=None, sampling=False, suffix=""):
    """`{sampling-}{data}-{tag-}gen{N}-{model}-{fmt}{suffix}.pkl`"""
    head = "sampling-" if sampling else ""
    tag_part = f"{tag}-" if tag else ""
    return f"{head}{data_name}-{tag_part}gen{num_gen}-{model_name}-{fmt}{suffix}.pkl"


def result_path(split, stage, data_name, model_name, fmt, num_gen,
                device_map=None, sampling=False, suffix=""):
    """split·stage 규약에 맞는 산출물 전체 경로."""
    lay = _layout(split)
    tag = device_tag(device_map) if lay["tagged"] else None
    return os.path.join(
        stage_dir(split, stage),
        result_name(data_name, model_name, fmt, num_gen, tag, sampling, suffix),
    )


_VLLM_SUBDIRS = (
    "results/not_post_processed", "results/post_processed",
    "results/double_loop", "results/single_kig",
    "dev_results/not_post_processed", "dev_results/post_processed",
    "ablation_results/posted_results",
)


def ensure_dirs():
    """산출물 디렉터리를 미리 만들어 둔다. 첫 실행 전에 한 번 호출하면 된다."""
    for sub in _VLLM_SUBDIRS:
        os.makedirs(os.path.join(VLLM_OUT, *sub.split("/")), exist_ok=True)


if __name__ == "__main__":
    for k in ("REPO_ROOT", "VLLM_OUT", "DPR_DATA"):
        print(f"{k:12s} = {globals()[k]}")
