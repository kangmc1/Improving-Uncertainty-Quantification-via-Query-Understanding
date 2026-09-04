# validation data를 사용해서 threshold를 체크하자.

import paths
import models
import utils_vllm
import pandas as pd
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='llama3.1_8b_it')
parser.add_argument("--num_gen", type=int, default=5)
parser.add_argument("--data_name", type=str, default="triviaqa") # nq, trivia, webq
parser.add_argument("--format", type=str, default='reflect') # direct, reflect
parser.add_argument("--device_map", type=str, default="2,3") # "0,1,2,3"
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

import os   
dmap = args.device_map
os.environ['CUDA_VISIBLE_DEVICES'] = dmap

import numpy as np


print("==="*20)
print("file name: threshold_save.py")
print("num_gen: ", args.num_gen)
print("model_name: ", args.model_name)
print("data_name: ", args.data_name)
print("format: ", args.format)
print("==="*20)


repo = models.repo(args.model_name)
    
device_path = paths.device_tag(args.device_map)


path_g = paths.result_path("dev", "scored", args.data_name, args.model_name,
                           args.format, args.num_gen, device_map=args.device_map)
path_s = paths.result_path("dev", "scored", args.data_name, args.model_name,
                           args.format, args.num_gen, device_map=args.device_map,
                           sampling=True)

with open(path_g, 'rb') as f:
    data_g = pickle.load(f)
with open(path_s, 'rb') as g:
    data_s = pickle.load(g)


df_g = pd.DataFrame(data_g)
df_s = pd.DataFrame(data_s)

df_merge = pd.merge(df_s, df_g, how='inner', on='question_id')

# lnse 데이터 post processing (nan => zero)
df_merge['lnse'] = df_merge['lnse'].apply(lambda x: 0.0 if np.isnan(x) else x)

# sar 데이터 post processing
df_merge['sar'] = df_merge['sar'].apply(lambda x: x[0])
df_merge['sar'] = df_merge['sar'].apply(lambda x: x.item())

df_merge['correct'] = (df_merge['rougel_score'] > 0.3).astype('int')
df_merge['accuracy'] = df_merge['correct'].mean()

# 1~99% 분위수로 탐색 (utils_vllm 기본값은 10~90%)
cut_range = np.arange(0.01, 0.99, 0.01)
thresholds = {}
for metric in ('pe', 'lnpe', 'lnse', 'sar'):
    value, n_sep = utils_vllm.th_vals(df_merge, metric, cut_range=cut_range)
    thresholds[metric] = value
    print(f"{metric:5s} threshold = {value:.4g}   (TP-FP = {n_sep})")

print()
print("utils_vllm.load_threshold 의 표에 아래 순서로 넣는다: [pe, lnpe, lnse, sar]")
print(f"  {[round(float(thresholds[m]), 4) for m in ('pe', 'lnpe', 'lnse', 'sar')]}")