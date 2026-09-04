# sampling 생성 결과 후처리 + 불확실성(PE / LNPE / LNSE / SAR) 계산.
#   --split test     : results/{not_post_processed -> post_processed}
#   --split dev      : dev_results/{not_post_processed -> post_processed}
#   --split ablation : ablation_results/{. -> posted_results}
# (예전의 sampling_eval_dev.py / sampling_eval_vllm_ablation.py 를 흡수했다.)

import paths
import argparse
import os
import pickle
from tqdm import tqdm

import common
import models
import uq_vllm

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='llama3.1-8b-it',
                    choices=models.names())
parser.add_argument("--num_gen", type=int, default=5)
parser.add_argument("--data_name", type=str, default="triviaqa")  # nq, triviaqa, webq
parser.add_argument("--format", type=str, default='direct')  # direct, reflect2, rec, rag, paraphrase, para-under
parser.add_argument("--device_map", type=str, default="1")  # "0,1,2,3" 조합 (ex: "0,1")
parser.add_argument("--split", type=str, default="test", choices=paths.SPLITS)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map

import torch

print("===" * 20)
print("file name: sampling_eval_vllm.py")
print("split     : ", args.split)
print("model_name: ", args.model_name)
print("data_name : ", args.data_name)
print("format    : ", args.format)
print("===" * 20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

from sentence_transformers.cross_encoder import CrossEncoder

cross_name = 'cross-encoder/stsb-roberta-large'
measure_model = CrossEncoder(model_name=cross_name, num_labels=1)

# 생성 결과 로딩
# keys: question_id, question, answer, formatted_input,
#       output_text_list, output_tokens_list, output_decodes_list, output_logprobs_list
in_path = paths.result_path(args.split, "raw", args.data_name, args.model_name,
                            args.format, args.num_gen,
                            device_map=args.device_map, sampling=True)
with open(in_path, 'rb') as f:
    dataset = pickle.load(f)

from transformers import AutoTokenizer

repo = models.repo(args.model_name)
patterns = models.answer_patterns(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
end_tokens = models.end_tokens(args.model_name, tokenizer)

# 데이터 길이 조정용
max_len = 128

nli_model = uq_vllm.ClassifyWrapper(device=device)
nli_model.model.to(device)

# cleanings and uq computation
with torch.inference_mode():
    for idx, data in enumerate(tqdm(dataset)):
        tokens_list, word_list, logprobs_list = [], [], []
        for i in range(len(data['output_decodes_list'])):
            tokens = data['output_tokens_list'][i]
            words = data['output_decodes_list'][i]
            logprobs = data['output_logprobs_list'][i]
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
            tokens_list.append(cleaned_tokens)
            word_list.append(cleaned_word)
            logprobs_list.append(cleaned_logprobs)

        similarity_dict = {'semantic_set_ids': uq_vllm._create_semantic_sets(
            nli_model.create_sim_mat_batched(data['question'], word_list))}

        pe = uq_vllm.pe(logprobs_list)
        lnpe = uq_vllm.lnpe_plus(logprobs_list)
        lnse = uq_vllm.se(logprobs_list, similarity_dict, 'ln', args.num_gen)
        token_importance = uq_vllm.get_tokenwise_importance(
            tokens_list, data['question'], measure_model, tokenizer)
        sentence_sim = uq_vllm.get_sentence_similarities(
            word_list, data['question'], measure_model)
        sar = uq_vllm.sar(logprobs_list, token_importance, sentence_sim,
                          t=0.001, num_generation=args.num_gen)

        dataset[idx].update({'cleaned_tokens_list': tokens_list,
                             'cleaned_word_list': word_list,
                             'cleaned_logprobs_list': logprobs_list,
                             'pe': pe,
                             'lnpe': lnpe,
                             'lnse': lnse,
                             'sar': sar})

out_path = paths.result_path(args.split, "scored", args.data_name, args.model_name,
                             args.format, args.num_gen,
                             device_map=args.device_map, sampling=True)
with open(out_path, 'wb') as out:
    pickle.dump(dataset, out)
print("saved:", out_path)
