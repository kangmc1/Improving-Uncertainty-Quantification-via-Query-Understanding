# sampling 생성 (질문당 답변 --num_gen 개).
#   --split test     -> results/not_post_processed/
#   --split ablation -> ablation_results/            (파일명에 device tag 없음)
# (예전의 sampling_vllm_ablation.py 를 흡수했다. dev split 은 데이터 소스가 달라
#  sampling_vllm_dev.py 로 분리되어 있다.)

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
print("file name: sampling_vllm.py")
print("split     : ", args.split)
print("model_name: ", args.model_name)
print("data_name : ", args.data_name)
print("format    : ", args.format)
print("===" * 20)

repo = models.repo(args.model_name)
temp = models.temperature(args.model_name)
n_gpu = len(args.device_map.split(','))

llm = LLM(model=repo, dtype=torch.bfloat16, tensor_parallel_size=n_gpu,
          enforce_eager=True, gpu_memory_utilization=0.95, trust_remote_code=True)
tokenizer = llm.get_tokenizer()

data = data_loader_vllm.data_loading(args.data_name, args.format)
dataset = data_loader_vllm.prompting_data_generation(data, tokenizer, args.format)

input_list = [dataset[i]['formatted_input'] for i in range(len(dataset))]

# ablation 실험은 원래부터 2048 로 돌렸다. 결과 재현을 위해 그대로 둔다.
max_tokens = 2048 if args.split == "ablation" else 4096
sampling_params = SamplingParams(temperature=temp, top_p=0.9, max_tokens=max_tokens,
                                 logprobs=1, n=args.num_gen)
outputs = llm.generate(input_list, sampling_params)


def output_returns(vllm_output):
    """vLLM 출력 하나에서 text / tokens / decodes / logprobs 를 뽑는다."""
    output_text = vllm_output.text
    output_tokens = list(vllm_output.token_ids)
    output_decodes = [vllm_output.logprobs[num][output_tokens[num]].decoded_token
                      for num in range(len(output_tokens))]
    output_logprob = [vllm_output.logprobs[num][output_tokens[num]].logprob
                      for num in range(len(output_tokens))]
    return output_text, output_tokens, output_decodes, output_logprob


for i in range(len(outputs)):
    elements = outputs[i].outputs
    output_text_list, output_tokens_list = [], []
    output_decodes_list, output_logprob_list = [], []
    for output in elements:
        output_text, output_tokens, output_decodes, output_logprob = output_returns(output)
        output_text_list.append(output_text)
        output_tokens_list.append(output_tokens)
        output_decodes_list.append(output_decodes)
        output_logprob_list.append(output_logprob)

    # 분석 결과 logprob 에 -1 을 곱해 nll 로 저장한다.
    output_nll = [[val * (-1) for val in value] for value in output_logprob_list]
    dataset[i].update({'output_text_list': output_text_list,
                       'output_tokens_list': output_tokens_list,
                       'output_decodes_list': output_decodes_list,
                       'output_logprobs_list': output_nll})

out_path = paths.result_path(args.split, "raw", args.data_name, args.model_name,
                             args.format, args.num_gen,
                             device_map=args.device_map, sampling=True)
with open(out_path, 'wb') as out:
    pickle.dump(dataset, out)
print("saved:", out_path)
