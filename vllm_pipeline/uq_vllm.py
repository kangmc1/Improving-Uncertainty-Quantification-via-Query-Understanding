import torch, math
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CONTRADICT, NEUTRAL, AGREE = 0, 1, 2

# response similarity calculation
class ClassifyWrapper():
    def __init__(self, model_name='microsoft/deberta-large-mnli', device='cuda:2') -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        pass

    @torch.no_grad()
    def _batch_pred(self, sen_1: list, sen_2: list, max_batch_size=128):
        inputs = [_[0] + ' [SEP] ' + _[1] for _ in zip(sen_1, sen_2)]
        inputs = self.tokenizer(inputs, padding=True, truncation=True, max_length=1024)
        input_ids = torch.tensor(inputs['input_ids']).to(self.model.device)
        attention_mask = torch.tensor(inputs['attention_mask']).to(self.model.device)
        logits = []
        for st in range(0, len(input_ids), max_batch_size):
            ed = min(st + max_batch_size, len(input_ids))
            logits.append(self.model(input_ids=input_ids[st:ed],
                                attention_mask=attention_mask[st:ed])['logits'])
        return torch.cat(logits, dim=0)

    @torch.no_grad()
    def create_sim_mat_batched(self, question, answers):
        unique_ans = sorted(list(set(answers)))
        semantic_set_ids = {ans: i for i, ans in enumerate(unique_ans)}
        _rev_mapping = semantic_set_ids.copy()
        sim_mat_batch = torch.zeros((len(unique_ans), len(unique_ans),3))
        anss_1, anss_2, indices = [], [], []
        for i, ans_i in enumerate(unique_ans):
            for j, ans_j in enumerate(unique_ans):
                if i == j: continue
                anss_1.append(f"{question} {ans_i}")
                anss_2.append(f"{question} {ans_j}")
                indices.append((i,j))
        if len(indices) > 0:
            sim_mat_batch_flat = self._batch_pred(anss_1, anss_2)
            for _, (i,j) in enumerate(indices):
                sim_mat_batch[i,j] = sim_mat_batch_flat[_]
        return dict(
            mapping = [_rev_mapping[_] for _ in answers],
            sim_mat = sim_mat_batch
        )

    @torch.no_grad()
    def _pred(self, sen_1: str, sen_2: str):
        input = sen_1 + ' [SEP] ' + sen_2
        input_ids = self.tokenizer.encode(input, return_tensors='pt').to(self.model.device)

        logits = self.model(input_ids)['logits']
        # logits: [Contradiction, neutral, entailment]
        return logits

    @torch.no_grad()
    def pred_qa(self, question:str, ans_1:str, ans_2:str):
        return self._pred(f"{question} {ans_1}", f'{question} {ans_2}')

    @torch.no_grad()
    def _compare(self, question:str, ans_1:str, ans_2:str):
        pred_1 = self._pred(f"{question} {ans_1}", f'{question} {ans_2}')
        pred_2 = self._pred(f"{question} {ans_2}", f'{question} {ans_1}')
        preds = torch.concat([pred_1, pred_2], 0)

        deberta_prediction = 0 if preds.argmax(1).min() == 0 else 1
        return {'deberta_prediction': deberta_prediction,
                'prob': torch.softmax(preds,1).mean(0).cpu(),
                'pred': preds.cpu()
                }

def _create_semantic_sets(sample):
    generated_texts = sample['mapping']
    sim_mat = sample['sim_mat'].argmax(axis=-1)
    # unique_ans is also a list of integers.
    unique_generated_texts = sorted(list(set(generated_texts)))
    semantic_set_ids = {ans: i for i, ans in enumerate(unique_generated_texts)} # one id for each exact-match answer
    for i, ans_i in enumerate(unique_generated_texts):
        for j, ans_j in enumerate(unique_generated_texts[i+1:], i+1):
            if min(sim_mat[ans_i,ans_j], sim_mat[ans_j,ans_i]) > CONTRADICT:
                semantic_set_ids[ans_j] = semantic_set_ids[ans_i]

    list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
    # map according to the order of appearance
    _map = defaultdict(int)
    ret = []
    for i, ans in enumerate(list_of_semantic_set_ids):
        if ans not in _map:
            _map[ans] = len(_map)
        ret.append(_map[ans])
    return ret

# 질문 텍스트, 생성응답 이렇게 두가지 데이터가 필요
@torch.no_grad()
def similarities(device, question, responses, model, tokenizer, id):
    _id = id
    result_dict = {}
    
    answer_list_1, answer_list_2, inputs = [], [], []
    semantic_set_ids = {}
    
    prompt = question
    generated_texts = responses
    unique_generated_texts = list(set(generated_texts))
    
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index
        
    if len(unique_generated_texts) > 1:
        for i, reference_answer in enumerate(unique_generated_texts):
            for j in range(i+1, len(unique_generated_texts)):
                answer_list_1.append(unique_generated_texts[i])
                answer_list_2.append(unique_generated_texts[j])
                
                qa_1 = prompt + ' ' + unique_generated_texts[i]
                qa_2 = prompt + ' ' + unique_generated_texts[j]
                inputs = qa_1 + ' [SEP] ' + qa_2

                encoded_input = tokenizer.encode(inputs, padding=True, max_length=256, truncation=True)
                prediction = model(torch.tensor([encoded_input], device=device, requires_grad=False))['logits']
                predicted_label = torch.argmax(prediction.cpu(), dim=1)
                
                reverse_input = qa_2 + ' [SEP] ' + qa_1
                encoded_reverse_input = tokenizer.encode(reverse_input, padding=True, max_length=256, truncation=True)
                reverse_prediction = model(torch.tensor([encoded_reverse_input], device=device, requires_grad=False))['logits']
                reverse_predicted_label = torch.argmax(reverse_prediction.cpu(), dim=1)
                
                deberta_prediction = 1
                if 0 in predicted_label or 0 in reverse_predicted_label:
                    has_semantically_different_answers = True
                    deberta_prediction = 0

                else:
                    semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]
                    
    list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
    result_dict = {'semantic_set_ids' : list_of_semantic_set_ids}
    
    return result_dict
    
# UQ computation

# 1. Discrete semantic entropy => sum(pln(p))
from collections import Counter
import numpy as np

def dse(sem_set):
    val = 0
    try:
        sem_data = sem_set['semantic_set_ids'].tolist()
    except:
        sem_data = sem_set['semantic_set_ids']
             
    length = len(sem_data)
    count_dict = dict(Counter(sem_data))
    for key in count_dict:
        p = count_dict[key]/length
        plnp = -p*np.log(p)
        val += plnp
        
    return val

# 2.5 lnpe plus
def lnpe_plus(token_ll_list):
    val = 0
    for seq in token_ll_list:
        if len(seq) < 1:
            lnpe = 0.0
            val += lnpe
        else:
            lnpe = sum(seq)/len(seq)
            val += lnpe
            
    val = val/len(token_ll_list) # 생성 response 갯수만큼 나눠주기
    return val

def pe(token_ll_list):
    val = 0
    for seq in token_ll_list:
        if len(seq) < 1:
            lnpe = 0.0
            val += lnpe
        else:
            lnpe = sum(seq)
            val += lnpe
            
    val = val/len(token_ll_list) # 생성 response 갯수만큼 나눠주기
    return val

# 3. SE, ln_SE
def se(token_lls_data, similarity, normality, num_generation=None):
    llh_shift = torch.tensor(5.0)
    new_likelihoods = []
    if num_generation is not None:
        new_likelihoods.append({'token_wise_entropy' : token_lls_data[:num_generation],
                                'semantic_set_ids': similarity['semantic_set_ids'][:num_generation]})
    
    likelihoods = new_likelihoods
    scores = []
    
    for sample_idx, likeli in enumerate(likelihoods):
        token_wise_entropy = likeli['token_wise_entropy']
        if normality == 'ln':
            # mean의 경우 ent값이 하나라도 nan이 있다면 에러가 뜰 수 있음
            ents_for_tensors = []
            for ent in token_wise_entropy:
                entropy = np.mean(ent)
                if math.isnan(entropy):
                    entropy = 0.0
                ents_for_tensors.append(entropy)
            gen_entropy = torch.tensor(ents_for_tensors).float().cpu()
        elif normality == 'sum':
            gen_entropy = torch.tensor([np.sum(ent) for ent in token_wise_entropy]).float().cpu()
            
        semantic_set_ids = torch.tensor(likeli['semantic_set_ids']).to(gen_entropy.device)
        semantic_cluster_entropy = []
        for semantic_id in torch.unique(semantic_set_ids):
            semantic_cluster_entropy.append(torch.logsumexp(-1 * gen_entropy[semantic_set_ids == semantic_id], dim=0))
        semantic_cluster_entropy = torch.tensor(semantic_cluster_entropy) - llh_shift
        semantic_cluster_entropy = - torch.sum(semantic_cluster_entropy, dim=0) / torch.tensor(
            semantic_cluster_entropy.shape[0])
        scores.append(torch.mean(semantic_cluster_entropy))
    
    score = scores[0]
    return score.item()

# 4. SAR

# token importance
@torch.no_grad()
def get_tokenwise_importance(token_responses, question, measure_model, tokenizer):
    token_importance_list = []
    for i in range(len(token_responses)):
        encoded_question = tokenizer.encode(question, add_special_tokens=False)
        encoded_answer = token_responses[i]
        decoded_total = tokenizer.decode(encoded_question + encoded_answer)
        token_importance = []
        for token in encoded_answer:
            sim_to_orig = measure_model.predict([decoded_total,
                                                 decoded_total.replace(
                                                     tokenizer.decode(token, skip_special_tokens=True),
                                                     '')]) 
        
            token_importance.append(1-torch.tensor(sim_to_orig))
            
        token_importance = torch.tensor(token_importance).reshape(-1)
        token_importance_list.append(token_importance)
    
    return token_importance_list

# sentence similarities
# responses => cleaned_generated_text_list
@torch.no_grad()
def get_sentence_similarities(responses, question, measure_model):
    similarity_list = []
    similarities = {}
    for i in range(len(responses)):
        similarities[i] = []
        
    for i in range(len(responses)):
        for j in range(i+1, len(responses)):
            gen_i = question + responses[i]
            gen_j = question + responses[j]
            similarity_i_j = measure_model.predict([gen_i, gen_j])
            similarities[i].append(similarity_i_j)
            similarities[j].append(similarity_i_j)
    
    similarity_list.append(similarities)        
    return similarity_list

def token_sar(token_lls, token_importance):
    likelihoods = [token_lls]
    scores = []
    error_count = 0
    for sample_idx, likeli in enumerate(likelihoods):
        gen_scores = []
        gen_token_wise_entropy = likeli
        for k in range(len(gen_token_wise_entropy)):
            token_wise_entropy = gen_token_wise_entropy[k].float()
            importance = token_importance[sample_idx * len(gen_token_wise_entropy) + k]
            if len(importance) == len(token_wise_entropy):
                weighted_score = ((importance / importance.sum()) * token_wise_entropy)
                gen_scores.append(torch.tensor(weighted_score).sum())
            else:
                error_count += 1
                gen_scores.append(0.0)

        gen_scores = torch.tensor(gen_scores)

        scores.append(gen_scores.mean())
        
    if error_count != 0:
        print(f'Error count of token sar: {error_count}')
        
    return scores

def sentence_sar(token_lls, sentence_sims, t=0.001):
    likelihoods = [token_lls]
    sentence_similarities = sentence_sims
    scores = []
    error_count = 0

    def semantic_weighted_log(similarities, entropies, t):
        probs = torch.exp(-1 * entropies)
        weighted_entropy = []
        for idx, (prob, ent) in enumerate(zip(probs, entropies)):
            w_ent = - torch.log(
                prob + ((torch.tensor(similarities[idx]) / t) * torch.cat([probs[:idx], probs[idx + 1:]])).sum())
            weighted_entropy.append(w_ent)
        return torch.tensor(weighted_entropy)

    for sample_idx, likeli in enumerate(likelihoods):
        gen_scores = []
        gen_token_wise_entropy = likeli
        for k in range(len(gen_token_wise_entropy)):
            token_wise_entropy = gen_token_wise_entropy[k].float()
            gen_scores.append(torch.tensor(token_wise_entropy).sum())

        similarity = sentence_similarities[sample_idx]
        gen_scores = torch.tensor(gen_scores)
        gen_scores = semantic_weighted_log(similarity, gen_scores, t=t)

        scores.append(gen_scores.mean())
    
    if error_count != 0:
        print(f'Error count: {error_count}')
        
    return scores

def sar(token_lls_data, token_importance, sentence_similarities, t=0.001, num_generation=None):
    token_importance = token_importance
    sentence_similarities = sentence_similarities
    new_likelihoods = []
    num_of_generated = num_generation
    if num_generation is not None:
        new_likelihoods.append({'token_wise_entropy': token_lls_data[:num_generation]})
        
    likelihoods = new_likelihoods
    scores = []
    error_count = 0

    def semantic_weighted_log(similarities, entropies, t, num_generation=None):
        probs = torch.exp(-1 * entropies)
        weighted_entropy = []
        for idx, (prob, ent) in enumerate(zip(probs, entropies)):
            if num_generation is not None:
                if idx + 1 >= num_generation:
                    w_ent = - torch.log(
                        prob + ((torch.tensor(similarities[idx][:num_generation - 1]) / t) * probs[:idx]).sum())
                else:
                    w_ent = - torch.log(
                        prob + ((torch.tensor(similarities[idx][:num_generation - 1]) / t) * torch.cat(
                            [probs[:idx], probs[idx + 1:num_generation]])).sum())
            else:
                w_ent = - torch.log(
                    prob + ((torch.tensor(similarities[idx]) / t) * torch.cat([probs[:idx], probs[idx + 1:]])).sum())
            weighted_entropy.append(w_ent)
        return torch.tensor(weighted_entropy)

    for sample_idx, likeli in enumerate(likelihoods):
        gen_scores = []
        gen_token_wise_entropy = likeli['token_wise_entropy']
        for k in range(len(gen_token_wise_entropy)):
            token_wise_entropy = torch.tensor(gen_token_wise_entropy[k]).float()
            importance = token_importance[sample_idx * num_of_generated + k]

            if len(importance) == len(token_wise_entropy):
                weighted_score = ((importance / importance.sum()) * token_wise_entropy)
                gen_scores.append(torch.tensor(weighted_score, requires_grad=False).sum())
            else:                
                error_count += 1
                gen_scores.append(0.0)

        similarity = sentence_similarities[sample_idx]
        gen_scores = torch.tensor(gen_scores)
        if num_generation is None or num_generation > 1:
            gen_scores = semantic_weighted_log(similarity, gen_scores, t=t, num_generation=num_generation)

        scores.append(gen_scores.mean())
        
    if error_count != 0:
        print(f'Error count: {error_count}')
        
    return scores

def rougel_compute(rouge, prediction, references):
    rougel_val = 0.0
    for refer in references:
        rougel = rouge.compute(predictions=prediction, references=[refer])['rougeL']
        rougel_val = max(rougel_val, rougel)
        
    return rougel_val