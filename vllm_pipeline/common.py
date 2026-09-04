

def slice_start_finder(output_tokens, patterns):

    last_index = 0
    for pattern in patterns:
        pat_len = len(pattern)
        for i in range(len(output_tokens) - pat_len + 1):
            if output_tokens[i:i + pat_len] == pattern:
                end_idx = i + pat_len
                if end_idx > last_index:
                    last_index = end_idx
    return last_index


def slice_end_finder(lst, patterns):
    earliest_index = len(lst)
    for pattern in patterns:
        pat_len = len(pattern)
        for i in range(len(lst) - pat_len + 1):
            if lst[i:i + pat_len] == pattern:
                if i < earliest_index:
                    earliest_index = i
                break
    return earliest_index


_ROUGE = None


def _rouge():
    global _ROUGE
    if _ROUGE is None:
        import evaluate
        _ROUGE = evaluate.load("rouge")
    return _ROUGE


def _compute_max_rouge(example):
    rouge = _rouge()
    pred = example["prediction"]
    refs = example["references"]

    scores = [rouge.compute(predictions=[pred], references=[ref], use_stemmer=True)
              for ref in refs]
    return {key: max(score[key] for score in scores) for key in scores[0]}


def max_rouge_scores(predictions, references, num_proc=12, metric="rougeL"):
    """예측마다 여러 정답 후보와 비교해 최대 rouge 점수를 구한다.

    predictions: 문자열 리스트
    references : 예측마다의 정답 후보 리스트들
    반환       : `metric` 점수 리스트 (predictions 와 같은 순서)
    """
    from datasets import Dataset

    d_set = Dataset.from_dict({
        "prediction": list(predictions),
        "references": list(references),
    })
    return d_set.map(_compute_max_rouge, num_proc=num_proc)[metric]
