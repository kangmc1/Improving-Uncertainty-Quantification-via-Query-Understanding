"""
모델 레지스트리 — repo · temperature · 정답 슬라이싱 토큰 패턴의 단일 소스.
"""

MODELS = {
    "llama3.1-8b-it": {
        "repo": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "temperature": 0.6,
        "answer_patterns": [[16533, 933], [16533, 60]],
        "extra_end_tokens": [],
    },
    "qwen2.5-7b-it": {
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "temperature": 0.7,
        "answer_patterns": [[16141, 921], [16141, 60]],
        "extra_end_tokens": [],
    },
    "mistral-8b-it": {
        "repo": "mistralai/Ministral-8B-Instruct-2410",
        "temperature": 0.7,
        "answer_patterns": [[31106, 2820], [31106, 1093]],
        "extra_end_tokens": [],
    },
    "olmo2-7b-it": {
        "repo": "allenai/OLMo-2-1124-7B-Instruct",
        "temperature": 0.6,
        "answer_patterns": [[16533, 933], [16533, 60]],
        "extra_end_tokens": [],
    },
}


def names():
    return sorted(MODELS)


def spec(model_name):
    try:
        return MODELS[model_name]
    except KeyError:
        raise KeyError(
            f"알 수 없는 --model_name '{model_name}'. "
            f"사용 가능: {', '.join(names())}"
        ) from None


def repo(model_name):
    return spec(model_name)["repo"]


def temperature(model_name):
    return spec(model_name)["temperature"]


def answer_patterns(model_name):
    # 호출부에서 `output_tokens[i:i+n] == pattern` 으로 리스트 비교를 하므로
    # 튜플이 아니라 리스트 사본을 돌려준다.
    return [list(p) for p in spec(model_name)["answer_patterns"]]


def end_tokens(model_name, tokenizer):
    """eos + 모델별 추가 종료 토큰."""
    return [[tokenizer.eos_token_id]] + [list(t) for t in spec(model_name)["extra_end_tokens"]]


if __name__ == "__main__":
    for name in names():
        s = MODELS[name]
        print(f"{name:16s} {s['repo']:42s} temp={s['temperature']}  "
              f"patterns={s['answer_patterns']}  end+={s['extra_end_tokens']}")
