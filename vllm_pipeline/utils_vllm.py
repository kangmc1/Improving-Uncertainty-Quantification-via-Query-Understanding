# AUROC metric
from sklearn import metrics
from sklearn.metrics import roc_curve

def auroc(pd_data, correct_val, uq_value):
    score = 0
    score = metrics.roc_auc_score(1-pd_data[correct_val], pd_data[uq_value])
    return score

import numpy as np

# 개선된 threshold 구하는 코드
def optimal_threshold_max_tp_minus_fp(y_true, y_score, tie_break="max_tp"):
    """
    목적: threshold 이상에서 TP - FP (= n_f - n_t) 최대화
    동치: Accuracy 최대화, prevalence-가중 Youden 최대화
    y_true: {0(true),1(false)}  # 여기서 1이 'false' 클래스
    y_score: 클래스 1(false)의 점수/확률
    tie_break: 동률일 때 선택 규칙 {"max_tp","min_fp","first"}
    """
    y_true = np.asarray(y_true)
    P = int((y_true == 1).sum())  # false=1 (positive)
    N = int((y_true == 0).sum())  # true=0  (negative)
    T = P + N

    # sklearn: y_score >= thr -> positive(여기서는 false=1로 예측)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    TP = P * tpr
    FP = N * fpr
    obj = TP - FP  # 우리가 최대화할 값 (n_f - n_t)
    best = obj.max()
    cand = np.where(obj == best)[0]

    if len(cand) > 1:
        if tie_break == "max_tp":
            idx = cand[np.argmax(TP[cand])]
        elif tie_break == "min_fp":
            idx = cand[np.argmin(FP[cand])]
        else:
            idx = cand[0]
    else:
        idx = cand[0]

    thr = float(thresholds[idx])
    acc = float((TP[idx] + (N - FP[idx])) / T)         # (TP+TN)/T
    youden_weighted = float((P/T)*tpr[idx] - (N/T)*fpr[idx])

    return {
        "threshold": thr,
        "TP": int(round(TP[idx])),
        "FP": int(round(FP[idx])),
        "TPR": float(tpr[idx]),
        "FPR": float(fpr[idx]),
        "num_TP-FP": int(obj[idx]),
        "accuracy": acc,
        "youden_weighted": youden_weighted,
        "P": P, "N": N
    }

# threshold functions
def th_vals(total_data, uq_type, cut_range=None, verbose=False):
    """TP-FP(= acc0 - acc1) 를 최대화하는 분위수 지점을 임계값으로 돌려준다.

    cut_range 기본값은 10~90% 분위수. threshold_vllm.py 는 1~99% 를 넘겨 쓴다.
    """
    if cut_range is None:
        cut_range = np.arange(0.1, 0.90, 0.01)  # 90 - 10 %

    uq_data = total_data[[uq_type, 'correct']]
    acc0 = uq_data.loc[uq_data.correct == 0, [uq_type]]
    acc1 = uq_data.loc[uq_data.correct == 1, [uq_type]]

    cut_points = [uq_data[uq_type].quantile(point) for point in cut_range]  # 기준이 전체인 경우
    num_list = []
    for cut_point in cut_points:
        acc0_n = len(acc0[acc0[uq_type] >= cut_point])
        acc1_n = len(acc1[acc1[uq_type] >= cut_point])
        num_list.append(acc0_n - acc1_n)

    if verbose:
        print(num_list)
    max_num = max(num_list)
    return cut_points[num_list.index(max_num)], max_num

def load_threshold(data_name, model_name, generation_name):
    th_dict = {'d000' : [1.823e-1, 6.075e-2, 3.451, -8.231],
        'd001' : [1.287e-1, 5.151e-2, 3.436, -8.216],
        'd010' : [9.891e-3, 4.687e-5, 3.390, -8.258],
        'd011' : [0.000, 0.000, 3.390, -8.269],
        'd020' : [5.763e-1, 5.435e-2, 3.444, -8.195],
        'd021' : [2.997e-1, 1.268e-1, 3.476, -8.179],
        'd040' : [4.083e-1, 6.338e-2, 3.456, -8.171],
        'd041' : [3.489e-1, 1.595e-1, 3.559, -8.068],
        'd100' : [8.158e-1, 3.017e-1, 3.822, -7.945],
        'd101' : [7.518e-1, 3.002e-1, 3.917, -7.972],
        'd110' : [7.194e-1, 1.587e-1, 3.834, -8.132],
        'd111' : [6.634e-1, 1.292e-1, 3.519, -8.047],
        'd120' : [1.502, 5.386, 4.017, -7.690],
        'd121' : [1.196, 4.314e-1, 3.922, -7.800],
        'd140' : [1.2273, 2.539e-1, 3.867, -7.849],
        'd141' : [8.599e-1, 3.430e-1, 3.811, -7.880],
        'd200' : [4.020e-1, 1.363e-1, 3.536, -8.022],
        'd201' : [2.373e-1, 1.232e-1, 3.508, -8.172],
        'd210' : [1.919e-1, 9.137e-2, 3.481, -8.212],
        'd211' : [2.139e-1, 6.428e-2, 3.436, -8.053],
        'd220' : [1.008, 3.408e-1, 3.731, -7.943],
        'd221' : [4.875e-1, 9.425e-2, 3.673, -7.891],
        'd240' : [1.3787, 1.963e-1, 3.579, -7.847],
        'd241' : [6.653e-1, 3.681e-1, 3.761, -7.887],
    }
    
    if data_name == 'nq':
        fn = '0'
    elif data_name == 'triviaqa':
        fn = '1'
    elif data_name == 'webq':
        fn = '2'
        
    if model_name == 'llama3.1-8b-it':
        sn = '0'
    elif model_name == 'qwen2.5-7b-it':
        sn = '1'
    elif model_name == 'mistral-8b-it':
        sn = '2'
    elif model_name == 'olmo2-7b-it':
        sn = '4'  # 3 은 예전 gemma2-27b-it 자리. 기존 키와 어긋나지 않게 번호를 그대로 둔다.
    
    if generation_name == 'direct':
        tn = '0'
    elif generation_name == 'reflect2':
        tn = '1'
    
    return_keys = f'd{fn}{sn}{tn}'
    threshold = th_dict[return_keys]
    return threshold