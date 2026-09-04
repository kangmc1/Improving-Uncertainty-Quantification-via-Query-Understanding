# AUROC return code

import paths
import pickle
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='llama3.1-8b-it') 
parser.add_argument("--num_gen", type=str, default=5)
parser.add_argument("--data_name", type=str, default="triviaqa") # nq, webq, triviaqa
parser.add_argument("--format", type=str, default='direct') 
parser.add_argument("--device_map", type=str, default="0,1") 
parser.add_argument("--split", type=str, default="test", choices=("test", "ablation"))
args = parser.parse_args()

# post_processing 데이터로부터 sampling, greedy 데이터들을 뽑아와서 question_id랑 매칭시킨 후 AUROC score 계산

data_path_greedy = paths.result_path(args.split, "scored", args.data_name, args.model_name,
                                     args.format, args.num_gen, device_map=args.device_map)
data_path_sampling = paths.result_path(args.split, "scored", args.data_name, args.model_name,
                                       args.format, args.num_gen, device_map=args.device_map,
                                       sampling=True)

with open(data_path_greedy, 'rb') as f:
    data_greedy = pickle.load(f)
    
with open(data_path_sampling, 'rb') as g:
    data_sampling = pickle.load(g)
    

def auprc_cal(correct, uq):
    precision, recall, _ = precision_recall_curve(1-correct, uq)
    auprc = auc(recall, precision)
    return auprc


df_greedy = pd.DataFrame(data_greedy)
df_sampling = pd.DataFrame(data_sampling)
df_greedy['correct'] = (df_greedy['rougel_score'] > 0.5).astype('int')
df_greedy['accuracy'] = df_greedy['correct'].mean()

metric_list = ['pe', 'lnpe','lnse','sar']

print('accuracy: ', df_greedy['accuracy'][0])
print('============================================')
print('============================================')

df_merge = pd.concat([df_sampling, df_greedy], axis=1)
df_merge = df_merge.groupby(df_merge.index).first()

# lnse 데이터 post processing (nan => zero)
df_merge['lnse'] = df_merge['lnse'].apply(lambda x: 0.0 if np.isnan(x) else x)

# sar 데이터 post processing
df_merge['sar'] = df_merge['sar'].apply(lambda x: x[0])
df_merge['sar'] = df_merge['sar'].apply(lambda x: x.item())

for metric in metric_list:
    print('metric: ', metric)
    print('auroc:  ', metrics.roc_auc_score(1-df_merge['correct'], df_merge[metric]))
    print('==='*15)
    
rouge_score = df_merge['rougel_score'].mean()

# f1 score
import regex, string
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1(prediction, ground_truths):
    return max([f1_score(prediction, gt) for gt in ground_truths])

f1_val = 0
for idx, ans in enumerate(df_merge['answer']):
    f1_val += f1(df_merge['cleaned_word'][idx], ans)
    
f1_avg = f1_val / len(df_merge['answer'])

    
print('======================')
print('======================')
print(f'{args.format} rouge: {rouge_score}')
print(f'{args.format} f1: {f1_avg}')