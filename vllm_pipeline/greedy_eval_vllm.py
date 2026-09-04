
import paths
import argparse
import pickle
from tqdm import tqdm

import common
import models

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='llama3.1-8b-it',
                    choices=models.names())
parser.add_argument("--num_gen", type=int, default=5)
parser.add_argument("--data_name", type=str, default="nq")  # nq, triviaqa, webq
parser.add_argument("--format", type=str, default='direct')  # direct, reflect2, rec, rag, paraphrase, para-under
parser.add_argument("--device_map", type=str, default="2,3")  # "0,1,2,3" 조합 (ex: "0,1")
parser.add_argument("--split", type=str, default="test", choices=paths.SPLITS)
parser.add_argument("--num_proc", type=int, default=12)  # rouge 배치 병렬도
args = parser.parse_args()

print("===" * 20)
print("file name: greedy_eval_vllm.py")
print("split     : ", args.split)
print("model_name: ", args.model_name)
print("data_name : ", args.data_name)
print("format    : ", args.format)
print("===" * 20)

# 생성 결과 로딩
# dict keys: question_id, question, answer, formatted_input,
#            output_text, output_tokens, output_decodes, output_logprobs
in_path = paths.result_path(args.split, "raw", args.data_name, args.model_name,
                            args.format, args.num_gen, device_map=args.device_map)
with open(in_path, 'rb') as f:
    dataset = pickle.load(f)

from transformers import AutoTokenizer

repo = models.repo(args.model_name)
patterns = models.answer_patterns(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
end_tokens = models.end_tokens(args.model_name, tokenizer)

max_len = 128

for idx, data in enumerate(tqdm(dataset)):
    tokens, words, logprobs = data['output_tokens'], data['output_decodes'], data['output_logprobs']
    start_idx = common.slice_start_finder(tokens, patterns)
    end_idx = common.slice_end_finder(tokens, end_tokens)
    cleaned_tokens, cleaned_words, cleaned_logprobs = (tokens[start_idx:end_idx],
                                                       words[start_idx:end_idx],
                                                       logprobs[start_idx:end_idx])
    # 뒤에서 n개 추출
    if len(cleaned_tokens) > max_len:
        cleaned_tokens, cleaned_words, cleaned_logprobs = (cleaned_tokens[-max_len:],
                                                           cleaned_words[-max_len:],
                                                           cleaned_logprobs[-max_len:])
    cleaned_word = tokenizer.decode(cleaned_tokens, skip_special_tokens=True)
    dataset[idx].update({'cleaned_tokens': cleaned_tokens,
                         'cleaned_word': cleaned_word,
                         'cleaned_logprobs': cleaned_logprobs})

# 배치 rouge
results_rougel = common.max_rouge_scores(
    [d['cleaned_word'] for d in dataset],
    [d['answer'] for d in dataset],
    num_proc=args.num_proc,
)

for idx in range(len(dataset)):
    dataset[idx].update({'rougel_score': results_rougel[idx]})

out_path = paths.result_path(args.split, "scored", args.data_name, args.model_name,
                             args.format, args.num_gen, device_map=args.device_map)
with open(out_path, 'wb') as out:
    pickle.dump(dataset, out)
print("saved:", out_path)
