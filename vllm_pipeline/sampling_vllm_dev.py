# dev dataset용 sampling with vllm

import paths
import models
import argparse
import pickle
import data_loader_vllm
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='llama3.1_8b_it')
# 사용 가능한 이름은 models.py 의 MODELS 참고
parser.add_argument("--num_gen", type=int, default=5)
parser.add_argument("--data_name", type=str, default="triviaqa") # nq, triviaqa, webq
parser.add_argument("--format", type=str, default='seqg_2s')
parser.add_argument("--device_map", type=str, default="1") # "0,1,2,3" 조합으로 생성(ex: "0,1")
args = parser.parse_args()

import os
dmap = args.device_map
os.environ['CUDA_VISIBLE_DEVICES'] = dmap

import torch

from vllm import LLM, SamplingParams


print("==="*20)
print("file name: sampling_vllm_dev.py")
print("model_name: ", args.model_name)
print("data_name: ", args.data_name)
print("format_name: ", args.format)
print("==="*20)


repo = models.repo(args.model_name)
temp = models.temperature(args.model_name)
        
n_gpu = len(args.device_map.split(","))
llm = LLM(model=repo, dtype=torch.bfloat16, tensor_parallel_size=n_gpu, enforce_eager=True, gpu_memory_utilization=0.95, trust_remote_code=True)
tokenizer = llm.get_tokenizer()


path = f"{paths.DPR_DATA}/retriever/"
nq = "nq-dev.json"
trivia = "trivia-dev.json"
webq = "webq_mydev.pkl"

if 'nq' in args.data_name:
    with open(path + nq, 'r') as f:
        qadata = json.load(f)
        
elif 'trivia' in args.data_name:
    with open(path + trivia, 'r') as f:
        qadata = json.load(f)

elif 'webq' in args.data_name:
    with open(path + webq, 'rb') as f:
        qadata = pickle.load(f)

def ctx_data_collection(data, psg_num):
    # passage 데이터를 n개만큼 묶는 과정 + 그 과정을 통합하는 과정
    psg_list = []
    contexts = []
    for i in range(len(data)):
        psg = []
        n_ctx = len(data[i]['positive_ctxs'])
        if n_ctx >= psg_num:
            for j in range(psg_num):
                psg.append(data[i]['positive_ctxs'][j])
                
        elif n_ctx < psg_num:
            for j in range(n_ctx):
                psg.append(data[i]['positive_ctxs'][j])
            nums = psg_num - n_ctx
            for k in range(nums):
                psg.append(data[i]['hard_negative_ctxs'][k])
        psg_list.append(psg)
        
        pag = ''
        for passage in psg:
            pag += '\n'
            pag += passage['text']
        
        pag = pag if pag.endswith(".") else pag + "."
        contexts.append(pag)
    return contexts

def ctx_data_collection_webq(data, psg_num):
    contexts = []
    
    for i in range(len(data)):
        psg = f''
        for j in range(psg_num):
            psg += '\n'
            psg += data[i]['docs'][j]
            
        psg = psg if psg.endswith(".") else psg + "."
        contexts.append(psg)
    return contexts

def prompting_data_generation(dataset, tokenizer, format):
    if 'reflect' in format:
        prompt_list = [data_loader_vllm.reflection_prompting(data['question']) for data in tqdm(dataset)]
    elif 'direct' in format:
        prompt_list = [data_loader_vllm.original_prompting(data['question']) for data in tqdm(dataset)]
    elif 'rag' in format:
        if 'webq' in args.data_name:
            ctx_datas = ctx_data_collection_webq(dataset, 5)
        else:
            ctx_datas = ctx_data_collection(dataset, 5) # 3: num passages
        prompt_list = [data_loader_vllm.rag_prompting(data['question'], ctx_datas[index]) for index, data in enumerate(tqdm(dataset))]
    
    my_data_list = []
    for idx, data in enumerate(dataset):
        inst_prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt_list[idx]}], tokenize=False, add_generation_prompt=True)
        
        if 'rag' in format:
            my_data_dict = {'question_id' : str(args.data_name) + '_' + str(idx),
                            'question' : data['question'] if data['question'].endswith("?") else data['question'] + "?",
                            'answer' : data['answers'],
                            'formatted_input': inst_prompt,
                            'ctxs' : ctx_datas[idx]}
        
        else:
            my_data_dict = {'question_id' : str(args.data_name) + '_' + str(idx),
                            'question' : data['question'] if data['question'].endswith("?") else data['question'] + "?",
                            'answer' : data['answers'],
                            'formatted_input': inst_prompt}
            
        my_data_list.append(my_data_dict)        
    
    return my_data_list

dataset = prompting_data_generation(qadata, tokenizer, args.format)

input_list = []

for i in range(len(dataset)):
    input_list.append(dataset[i]['formatted_input'])

sampling_params = SamplingParams(temperature=temp, top_p=0.9, max_tokens=2048, logprobs=1, n=5)

outputs = llm.generate(input_list, sampling_params)

def output_returns(vllm_output):
    output_text = vllm_output.text
    output_tokens = list(vllm_output.token_ids)
    output_decodes = [vllm_output.logprobs[num][output_tokens[num]].decoded_token for num in range(len(output_tokens))]
    output_logprob = [vllm_output.logprobs[num][output_tokens[num]].logprob for num in range(len(output_tokens))]
    return output_text, output_tokens, output_decodes, output_logprob

for i in range(len(outputs)):
    elements = outputs[i].outputs
    output_text_list = []
    output_tokens_list = []
    output_decodes_list = []
    output_logprob_list = []
    for output in elements:
        output_text, output_tokens, output_decodes, output_logprob = output_returns(output)
        output_text_list.append(output_text)
        output_tokens_list.append(output_tokens)
        output_decodes_list.append(output_decodes)
        output_logprob_list.append(output_logprob)
        
    output_nll = [[val*(-1) for val in value] for value in output_logprob_list]
    # 분석결과 logprob에다가 싹다 -1을 곱해줘야됨
    dataset[i].update({'output_text_list' : output_text_list,
                       'output_tokens_list' : output_tokens_list,
                       'output_decodes_list' : output_decodes_list,
                       'output_logprobs_list' : output_nll})
    
# dataset[i]의 요소들
# question_id, question, answer, formatted_input, contexts(optional)
# output_text, output_tokens, output_decodes, output_logprobs


device_path = paths.device_tag(dmap)
    

path = paths.result_path("dev", "raw", args.data_name, args.model_name,
                         args.format, args.num_gen, device_map=dmap,
                         sampling=True)
with open(path, 'wb') as out:
    pickle.dump(dataset, out)