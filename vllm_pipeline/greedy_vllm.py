
import paths
import argparse
import os
import pickle

import models

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='llama3.1-8b-it',
                    choices=models.names())
parser.add_argument("--num_gen", type=int, default=5)
parser.add_argument("--data_name", type=str, default="triviaqa")  # nq, triviaqa, webq
parser.add_argument("--format", type=str, default='direct')  # direct, reflect2, rec, rag, paraphrase, para-under
parser.add_argument("--device_map", type=str, default="0,1")  # "0,1,2,3" 조합 (ex: "0,1")
parser.add_argument("--split", type=str, default="test", choices=("test", "ablation"))
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map

import torch
from vllm import LLM, SamplingParams

import data_loader_vllm

print("===" * 20)
print("file name: greedy_vllm.py")
print("split     : ", args.split)
print("model_name: ", args.model_name)
print("data_name : ", args.data_name)
print("format    : ", args.format)
print(f"using {len(args.device_map.split(','))} GPUs: {args.device_map}")
print("===" * 20)

repo = models.repo(args.model_name)
temp = models.temperature(args.model_name)
n_gpu = len(args.device_map.split(','))

# tensor_parallel_size 는 보유 GPU 수에 맞춰 조정
llm = LLM(model=repo, dtype=torch.bfloat16, tensor_parallel_size=n_gpu,
          trust_remote_code=True)
tokenizer = llm.get_tokenizer()

data = data_loader_vllm.data_loading(args.data_name, args.format)
dataset = data_loader_vllm.prompting_data_generation(data, tokenizer, args.format)

input_list = [dataset[i]['formatted_input'] for i in range(len(dataset))]

sampling_params = SamplingParams(temperature=temp, max_tokens=4096, logprobs=1)
outputs = llm.generate(input_list, sampling_params)

for i in range(len(outputs)):
    elements = outputs[i].outputs
    output_text = elements[0].text
    output_tokens = list(elements[0].token_ids)
    output_decodes = [elements[0].logprobs[num][output_tokens[num]].decoded_token
                      for num in range(len(output_tokens))]
    output_logprobs = [elements[0].logprobs[num][output_tokens[num]].logprob
                       for num in range(len(output_tokens))]
    dataset[i].update({'output_text': output_text,
                       'output_tokens': output_tokens,
                       'output_decodes': output_decodes,
                       'output_logprobs': output_logprobs})

# dataset[i]의 키:
# question_id, question, answer, formatted_input, contexts(optional),
# output_text, output_tokens, output_decodes, output_logprobs

out_path = paths.result_path(args.split, "raw", args.data_name, args.model_name,
                             args.format, args.num_gen, device_map=args.device_map)
with open(out_path, 'wb') as out:
    pickle.dump(dataset, out)
print("saved:", out_path)
