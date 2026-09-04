# 불확실성과 무관하게 모든 질문을 rec/rag 로 재생성하는 비교군.
# 생성과 RougeL 채점을 한 번에 처리한다.

import paths
import argparse
import pickle
import data_loader_vllm
import common
import models

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='llama3.1-8b-it')
# 사용 가능한 이름은 models.py 의 MODELS 참고
parser.add_argument("--num_gen", type=int, default=5)
parser.add_argument("--data_name", type=str, default="triviaqa") # nq, triviaqa, webq
parser.add_argument("--format", type=str, default='rec') # rec, rag
parser.add_argument("--device_map", type=str, default="0,1") # "0,1,2,3" 조합으로 생성(ex: "0,1")
# parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_proc", type=int, default=12)  # rouge 배치 병렬도
args = parser.parse_args()

import os
dmap = args.device_map
os.environ['CUDA_VISIBLE_DEVICES'] = dmap

import torch

from vllm import LLM, SamplingParams

print("==="*20)
print("file name: gen_kig_vllm.py")
print("model_name: ", args.model_name)
print("data_name: ", args.data_name)
print("format_name: ", args.format)
print("==="*20)

repo = models.repo(args.model_name)
temp = models.temperature(args.model_name)

n_gpu = len(args.device_map.split(','))

llm = LLM(model=repo, dtype=torch.bfloat16, tensor_parallel_size=n_gpu) 
tokenizer = llm.get_tokenizer()
data = data_loader_vllm.data_loading(args.data_name, args.format)
dataset = data_loader_vllm.prompting_data_generation(data, tokenizer, args.format)

input_list = []

for i in range(len(dataset)):
    input_list.append(dataset[i]['formatted_input'])

sampling_params = SamplingParams(temperature=temp, max_tokens=4096, logprobs=1)

outputs = llm.generate(input_list, sampling_params)

patterns = models.answer_patterns(args.model_name)

end_tokens = models.end_tokens(args.model_name, tokenizer)

max_len = 128

for i in range(len(outputs)):
    elements = outputs[i].outputs
    output_text = elements[0].text
    output_tokens = list(elements[0].token_ids)
    output_decodes = [elements[0].logprobs[num][output_tokens[num]].decoded_token for num in range(len(output_tokens))]
    start_idx = common.slice_start_finder(output_tokens, patterns)
    end_idx = common.slice_end_finder(output_tokens, end_tokens)
    cleaned_tokens = output_tokens[start_idx:end_idx]
    if len(cleaned_tokens) > max_len:
        cleaned_tokens = cleaned_tokens[-max_len:]
    cleaned_word = tokenizer.decode(cleaned_tokens, skip_special_tokens=True)
    dataset[i].update({'output_text': output_text,
                       'output_tokens': output_tokens,
                       'output_decodes': output_decodes,
                       'cleaned_tokens' : cleaned_tokens,
                       'cleaned_word' : cleaned_word})
    
results_rougel = common.max_rouge_scores(
    [d['cleaned_word'] for d in dataset],
    [d['answer'] for d in dataset],
    num_proc=args.num_proc,
)

for idx, data_dict in enumerate(dataset):
    dataset[idx].update({'rougel_score' : results_rougel[idx]})

device_path = paths.device_tag(dmap)


path = os.path.join(
    paths.VLLM_OUT, "results", "single_kig",
    paths.result_name(args.data_name, args.model_name, args.format, args.num_gen,
                      tag=device_path),
)
with open(path, 'wb') as out:
    pickle.dump(dataset, out)