
import paths
import data_loader_vllm
import pickle
import torch
import argparse
import utils_vllm
import common
import models

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='llama3.1-8b-it')
# 사용 가능한 이름은 models.py 의 MODELS 참고
parser.add_argument("--num_gen", type=int, default=5)
parser.add_argument("--data_name", type=str, default="triviaqa") # nq, triviaqa, webq
parser.add_argument("--format", type=str, default='direct') # direct, reflect
parser.add_argument("--format2", type=str, default='rec') # rag, rec
parser.add_argument("--look", type=str, default='pe') # pe, lnpe, lnse, sar, vote
parser.add_argument("--device_map", type=str, default="0,1")
parser.add_argument("--num_proc", type=int, default=24)  # rouge 배치 병렬도
args = parser.parse_args()

import os
dmap = args.device_map
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = dmap


device_path = paths.device_tag(dmap)

import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
    
repo = models.repo(args.model_name)
temp = models.temperature(args.model_name)
patterns = models.answer_patterns(args.model_name)

thresholds = utils_vllm.load_threshold(args.data_name, args.model_name, args.format)

pe, lnpe, se, sar = thresholds[0], thresholds[1], thresholds[2], thresholds[3]
if args.look == 'pe':
    threshold = pe
elif args.look == 'lnpe':
    threshold = lnpe
elif args.look == 'lnse':
    threshold = se
elif args.look == 'sar':
    threshold = sar
elif args.look == 'vote': # majority voting mechanism
    threshold = thresholds
    
print('pe, lnpe, se, sar: ', pe, lnpe, se, sar)

gp_path = paths.result_path("test", "scored", args.data_name, args.model_name,
                            args.format, args.num_gen, device_map=dmap)
sp_path = paths.result_path("test", "scored", args.data_name, args.model_name,
                            args.format, args.num_gen, device_map=dmap, sampling=True)

with open(gp_path, 'rb') as f:
    data_greedy = pickle.load(f)
with open(sp_path, 'rb') as g:
    data_sampling = pickle.load(g)
    
n_gpu = len(dmap.split(','))
llm = LLM(model=repo, dtype=torch.bfloat16, tensor_parallel_size=n_gpu, gpu_memory_utilization=0.95)
tokenizer = llm.get_tokenizer()

end_tokens = models.end_tokens(args.model_name, tokenizer)

# 필요데이터
# g: question_id, question, answer, cleaned_word, rougel_score
# s: question_id, pe, lnpe, lnse, sar

new_dataset = []


if args.look == 'one':
    for idx, data in enumerate(data_greedy):
        if data_sampling[idx][f'pe'] >= thresholds[0]:
            new_dataset.append(data)
elif args.look == 'two':
    for idx, data in enumerate(data_greedy):
        tf_list = [(data_sampling[idx]['pe'] >= thresholds[0]), (data_sampling[idx]['lnpe'] >= thresholds[1])]
        cnt = 0
        for t in tf_list:
            if t:
                cnt += 1
        if cnt >= 1:
            new_dataset.append(data)
elif args.look == 'three':
    for idx, data in enumerate(data_greedy):
        tf_list = [(data_sampling[idx]['pe'] >= thresholds[0]), (data_sampling[idx]['lnpe'] >= thresholds[1]), (data_sampling[idx]['lnse'] >= thresholds[2])]
        cnt = 0
        for t in tf_list:
            if t:
                cnt += 1
        if cnt >= 2:
            new_dataset.append(data)
else:
    for idx, data in enumerate(data_greedy):
        tf_list = [(data_sampling[idx]['pe'] >= thresholds[0]), (data_sampling[idx]['lnpe'] >= thresholds[1]), (data_sampling[idx]['lnse'] >= thresholds[2]), (data_sampling[idx]['sar'][0].item() >= thresholds[3])]
        cnt = 0
        for t in tf_list:
            if t:
                cnt += 1
        if cnt >= 2:
            new_dataset.append(data)
            

# 여기서 rag 데이터를 불러오고 reform_generation으로 프롬프팅
if 'rag' in args.format2:
    data_for_rag = data_loader_vllm.data_loading(args.data_name, args.format2)
else:
    data_for_rag = None

for idx, d in enumerate(new_dataset):
    new_dataset[idx].update({f'prompt_{args.format2}' : data_loader_vllm.reform_generation(d, tokenizer, args.format2, data_for_rag)})

inst_input_list = []
for dd in new_dataset:
    inst_input_list.append(dd[f'prompt_{args.format2}'])
    
sampling_params = SamplingParams(temperature=temp, max_tokens=2048, logprobs=1)
outputs = llm.generate(inst_input_list, sampling_params)



max_len = 128

for i in tqdm(range(len(outputs))):
    elements = outputs[i].outputs
    output_tokens = list(elements[0].token_ids)
    output_decodes = [elements[0].logprobs[num][output_tokens[num]].decoded_token for num in range(len(output_tokens))]
    start_idx = common.slice_start_finder(output_tokens, patterns)
    end_idx = common.slice_end_finder(output_tokens, end_tokens)
    clean_tokens, clean_words = output_tokens[start_idx:end_idx], output_decodes[start_idx:end_idx]
    if len(clean_tokens) > max_len:
        clean_tokens, clean_words = clean_tokens[-max_len:], clean_words[-max_len:]
    clean_word = tokenizer.decode(clean_tokens, skip_special_tokens=True)
    new_dataset[i].update({'clean_word' : clean_word})
    
results_rougel = common.max_rouge_scores(
    [d['clean_word'] for d in new_dataset],
    [d['answer'] for d in new_dataset],
    num_proc=args.num_proc,
)

q_id_list = []
for i in range(len(new_dataset)):
    q_id_list.append(new_dataset[i]['question_id'])
    
# by question_id, input new rouge
for idx, data in enumerate(tqdm(data_greedy)):
    if data['question_id'] in q_id_list:
        num = q_id_list.index(data['question_id'])
        data_greedy[idx].update({f'rougel_score_{args.look}' : results_rougel[num]})
    else:
        data_greedy[idx].update({f'rougel_score_{args.look}' : data['rougel_score']})
        
saves = os.path.join(
    paths.VLLM_OUT, "results", "double_loop",
    paths.result_name(args.data_name, args.model_name, args.format, args.num_gen,
                      tag=device_path, suffix=f"-{args.format2}-{args.look}"),
)
with open(saves, 'wb') as out:
    pickle.dump(data_greedy, out)