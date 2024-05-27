import os
import pdb
import random
from tqdm import tqdm
import numpy as np
import itertools
import torch
from model.main import T5SequentialRecommender
from model.modules.p5.notebooks.evaluate.metrics4rec import evaluate_all
import pickle
from model.utils import Trie, predict_outputs, prefix_allowed_tokens_fn
from transformers import AutoTokenizer
import json


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(filename, file):
    with open(filename, 'wb') as fOut:
        pickle.dump(file, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    return print('saving done!')


def ReadLineFromFile(path):
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


from collections import defaultdict

# domain = 'Beauty'
domain = 'Sports_and_Outdoors'
# domain = 'Toys_and_Games'
# domain = 'Yelp'
root_path = f'data/amazon/filtered/{domain}/rqid/'
iid_data = ReadLineFromFile(os.path.join(root_path, 'sequential_data.txt'))
interaction_count = {}
for sequence in iid_data:
    user, items = sequence.strip().split(' ', 1)
    items = items.split(' ')
    for item in items:
        if item in interaction_count.keys():
            interaction_count[item] += 1
        else:
            interaction_count[item] = 1

# gid_path = "sequential_data-Beauty-gid_253.json"
# sid_path = "sequential_data-Beauty-sid_949.json"

gid_path = "sequential_data-Sports-gid_302.json"
sid_path = "sequential_data-Sports-sid_1523.json"

# gid_path = "sequential_data_Toys-gid_257.json"
# sid_path = "sequential_data_Toys-sid_996.json"

# gid_path = "sequential_data_Yelp-gid-best-batch_126.json"
# sid_path = "sequential_data_Yelp-sid-768-batch_1364.json"

# gid_path = "sequential_data_Yelp-gid-best-batch-nodecay_110.json"
# sid_path = "sequential_data_Yelp-sid-768-batch-nodecay_1137.json"

# m_name = '0503-Beauty-both-s2023-3090'
# m_name = '0503-Beauty-both-s2024-3090'
# m_name = '0503-Beauty-both-s2025-3090'
# m_name = '0505-Beauty-both-s2023-a100'
# m_name = '0505-Beauty-both-s2024-a100'
# m_name = '0505-Beauty-both-s2025-a100'


# m_name = '0503-Sports-both-s2023-3090'
m_name = '0503-Sports-both-s2024-3090'
# m_name = '0503-Sports-both-s2025-3090'
# m_name = '0505-Sports-both-s2023-a100'
# m_name = '0505-Sports-both-s2024-a100'
# m_name = '0505-Sports-both-s2025-a100'


# m_name = '0510-Toys-both-s1'
# m_name = '0510-Toys-both-s2'
# m_name = '0510-Toys-both-s3'
# m_name = '0510-Toys-both-s4'
# m_name = '0510-Toys-both-s5'

# m_name = '0514-Yelp-both-s1'
# m_name = '0514-Yelp-both-s2'
# m_name = '0514-Yelp-both-s3'
# m_name = '0514-Yelp-both-s4'
# m_name = '0514-Yelp-both-s5'

# m_name = '0515-Yelp-both-nodecay-s1'
# m_name = '0515-Yelp-both-nodecay-s2'
# m_name = '0515-Yelp-both-nodecay-s3'
# m_name = '0515-Yelp-both-nodecay-s4'
# m_name = '0515-Yelp-both-nodecay-s5'



def scoring_f(x, score_type):
    if score_type == 'exp':
        return np.exp(-x / T1)
    elif score_type == 'log':
        return 1 / np.log2(x + 2)
    elif score_type == 'recip':
        return 1 / (x + 1)
    else:
        raise NotImplementedError


def reciprocal_fusion(candidates=[], roh=0.1):
    return np.mean(((1 / np.array(candidates)) ** roh)) ** (1 / roh)


def combine_f(candidates=[], combine_type='add', alpha=1):
    if combine_type == 'log':
        return np.sum(np.log2(np.array(candidates)))
    elif combine_type == 'recip':
        return reciprocal_fusion(candidates)
    else:
        return alpha * candidates[0] + candidates[1]


def ensemble_f(candidates=[], ensemble_type='add'):
    if ensemble_type == 'log':
        return np.sum(np.log2(np.array(candidates)))
    elif ensemble_type == 'log-mean':
        return np.mean(np.log2(np.array(candidates)))
    elif ensemble_type == 'recip':
        return reciprocal_fusion(candidates)
    elif ensemble_type == 'mean':
        return np.mean(candidates)
    else:
        return np.sum(candidates)


score_type = 'exp'
# score_type = 'log'
# score_type = 'recip'

combine_type = 'add'

# combine_type = 'log'
# combine_type = 'recip'

ensemble_type = 'add'
# ensemble_type = 'mean'
# ensemble_type = 'log'
# ensemble_type = 'log-mean'
# ensemble_type = 'recip'

do_per = False
roh = 0.1
min_thresh = 2
alpha = 4
k = 5
# k = 10


beam_size = 20
# T1 = 2
# T1 = 5
# T1 = 10
# T1 = 20
T1 = 40
# instance_num = 22363
instance_num = 35598
# instance_num = 19412
# instance_num = 30431
template_num = 10
mix_temp_num = 10

pred_path0 = f'saved/Recommender/{m_name}/prediction_list_{beam_size}_0'
pred0 = load_pickle(pred_path0)
pred_path1 = f'saved/Recommender/{m_name}/prediction_list_{beam_size}_1'
pred1 = load_pickle(pred_path1)
# pred_path2 = f'saved/Recommender/{m_name}/prediction_list_{beam_size}_2'
# pred2 = load_pickle(pred_path2)
# pred_path3 = f'saved/Recommender/{m_name}/prediction_list_{beam_size}_3'
# pred3 = load_pickle(pred_path3)

preds = []
preds.append(pred0)
preds.append(pred1)
# preds.append(pred2)
# preds.append(pred3)

ensemble_num = len(preds)
# ensemble_num = 2

iid_gid = load_json(os.path.join(root_path, 'rq-gid', gid_path))
iid_sid = load_json(os.path.join(root_path, 'rq-sid', sid_path))
gid_iid = defaultdict(list)
sid_iid = defaultdict(list)
idx_iid = defaultdict(list)
for key, v in iid_gid.items():
    text = 'item_<G>'
    for level, code in enumerate(v):
        text += f'<extra_g_{level}_{code}>'
    gid_iid[text] = key
    idx_iid[text] = key
for key, v in iid_sid.items():
    text = 'item_<S>'
    for level, code in enumerate(v):
        text += f'<extra_s_{level}_{code}>'
    sid_iid[text] = key
    idx_iid[text] = key

'''
개별 template prediction
'''
# [4, 22363, 10]
corr_arr = np.zeros((ensemble_num, instance_num, template_num, beam_size))
for e in range(ensemble_num):
    for i in tqdm(range(instance_num)):
        for t in range(template_num):
            corr_arr[e, i, t] = idx_iid[preds[e][template_num * i + t][0]] == np.array([idx_iid[p] for p in preds[e][template_num * i + t][1][:beam_size]])

hit_arr = np.sum(corr_arr[:, :, :mix_temp_num, :k], axis=-1)
ndcg_arr = np.sum(corr_arr[:, :, :mix_temp_num, :k] / np.log2(np.arange(2, k + 2)), axis=-1)
total_avg_hit = 0
total_avg_ndcg = 0
for e in range(ensemble_num):
    temp_avg_hit = 0
    for t in range(mix_temp_num):
        temp_hit = np.not_equal(hit_arr[e, :, t], 0).sum() / instance_num
        temp_avg_hit += temp_hit
        total_avg_hit += temp_hit
    #     print(f'hit@{k}_{e}_{t}: {temp_hit}')
    print(f'average hit@{k}_{e} ratio \n {temp_avg_hit / (mix_temp_num)}')

    temp_avg_ndcg = 0
    for t in range(mix_temp_num):
        temp_ndcg = ndcg_arr[e, :, t].sum() / instance_num
        temp_avg_ndcg += temp_ndcg
        total_avg_ndcg += temp_ndcg
    #     print(f'ndcg@{k}_{e}_{t}: {temp_ndcg}')
    print(f'average ndcg@{k}_{e} ratio \n {temp_avg_ndcg / (mix_temp_num)}')

print(f'total_avg_hit@{k} : \n {total_avg_hit / ensemble_num}')
print(f'total_avg_ndcg@{k} : \n {total_avg_ndcg / ensemble_num}')
h_all = np.not_equal(np.sum(np.sum(hit_arr, axis=-1), axis=0), 0.)
h_g_all = np.not_equal(np.sum(hit_arr[0], axis=-1), 0.)
h_s_all = np.not_equal(np.sum(hit_arr[1], axis=-1), 0.)
print('union')
print(h_all.sum() / instance_num)

# per
if do_per:
    per_gg = np.zeros([template_num, template_num])
    per_ss = np.zeros([template_num, template_num])
    per_gs = np.zeros([template_num, template_num])
    for a in range(template_num):
        for b in range(template_num):
            per_gg[a, b] = sum((hit_arr[0, :, a] - hit_arr[0, :, b]) > 0) / sum(hit_arr[0, :, a]) * 100
            per_ss[a, b] = sum((hit_arr[1, :, a] - hit_arr[1, :, b]) > 0) / sum(hit_arr[1, :, a]) * 100
            per_gs[a, b] = sum((hit_arr[0, :, a] - hit_arr[1, :, b]) > 0) / sum(hit_arr[0, :, a]) * 100
    import matplotlib.pyplot as plt
    import seaborn as sns

    title_font = {
        'fontsize': 40,
        'fontweight': 'bold'
    }
    plt.figure(figsize=(template_num, template_num))
    plt.title('CEID-CEID', fontdict=title_font)
    sns.set(font_scale=2)  # 아주 크게
    gg = sns.heatmap(data=per_gg, annot=True, annot_kws={"size": 18}, fmt='.1f', linewidths=.5, cmap='Reds', vmin=0, vmax=10)
    cbar = gg.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)

    plt.figure(figsize=(template_num, template_num))
    plt.title('SEID-SEID', fontdict=title_font)
    ss = sns.heatmap(data=per_ss, annot=True, annot_kws={"size": 18}, fmt='.1f', linewidths=.5, cmap='Reds', vmin=0, vmax=10)
    cbar = ss.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)

    # plt.figure(figsize=(template_num, template_num))
    # plt.title('CID-SID')
    # gs = sns.heatmap(data=per_gs, annot=True, annot_kws={"size": 20}, fmt='.1f', linewidths=.5, cmap='Blues', vmin=0, vmax=50)
    plt.show()

# print('chr')
# avg_chr = 0
# for e in range(2):
#     for t in range(mix_temp_num):
#         chr = (h_all - hit_arr[e, :, t]).sum() / h_all.sum() * 100
#         print(chr)
#         avg_chr += chr
# print(f'avg: {avg_chr / 20}')
#
# print('chr-g')
# avg_chr = 0
# for t in range(mix_temp_num):
#     chr = (h_g_all - hit_arr[0, :, t]).sum() / h_g_all.sum() * 100
#     print(chr)
#     avg_chr += chr
# print(f'avg: {avg_chr / 10}')
#
# print('chr-s')
# avg_chr = 0
#
# for t in range(mix_temp_num):
#     chr = (h_s_all - hit_arr[1, :, t]).sum() / h_s_all.sum() * 100
#     print(chr)
#     avg_chr += chr
# print(f'avg: {avg_chr / 10}')

'''
position score 기반 consensus prediction
'''
from collections import defaultdict

gt_dict = {}
# 등장 빈도수
freq_dict = {e: {i: {} for i in range(instance_num)} for e in range(ensemble_num)}
rank_dict = {e: {i: {} for i in range(instance_num)} for e in range(ensemble_num)}
R_dict = {e: {i: {} for i in range(instance_num)} for e in range(ensemble_num)}
C_dict = {e: {i: {} for i in range(instance_num)} for e in range(ensemble_num)}
RC_dict = {e: {i: {} for i in range(instance_num)} for e in range(ensemble_num)}

ER_dict = {i: {} for i in range(instance_num)}
EC_dict = {i: {} for i in range(instance_num)}
ERC_dict = {i: {} for i in range(instance_num)}

ER_k = []
EC_k = []
ERC_k = []
RC1_k = []
RC2_k = []

ER_corr_arr = np.zeros((instance_num, k))
EC_corr_arr = np.zeros((instance_num, k))
ERC_corr_arr = np.zeros((instance_num, k))
RC1_corr_arr = np.zeros((instance_num, k))
RC2_corr_arr = np.zeros((instance_num, k))

for i in tqdm(range(instance_num)):
    for e in range(ensemble_num):
        gt = idx_iid[preds[e][template_num * i][0]]
        gt_dict[i] = gt
        for t in range(mix_temp_num):
            pred_list = preds[e][template_num * i + t][1][:beam_size]
            for rank, pred in enumerate(pred_list):
                if idx_iid[pred] in freq_dict[e][i].keys():
                    freq_dict[e][i][idx_iid[pred]] += 1
                    rank_dict[e][i][idx_iid[pred]].append(rank)
                else:
                    freq_dict[e][i][idx_iid[pred]] = 1
                    rank_dict[e][i][idx_iid[pred]] = [rank]

        # consistency 계산
        for iid, rank_list in rank_dict[e][i].items():
            mean_rank = np.mean(rank_list)
            position_score = scoring_f(mean_rank, score_type)
            R_dict[e][i][iid] = position_score

            n = len(rank_list)
            if n < 2:
                var_x = beam_size ** 2
            else:
                var_x = np.sum((np.array(rank_list) - mean_rank) ** 2) / (n - 1)
            sstd = np.sqrt(var_x)
            consistency_score = scoring_f(sstd, score_type)
            C_dict[e][i][iid] = consistency_score
            RC_dict[e][i][iid] = combine_f([position_score, consistency_score], combine_type, alpha)

for i in tqdm(range(instance_num)):
    temp_r = {}
    temp_c = {}
    temp_rc = {}
    temp_1 = {}
    temp_2 = {}
    rank_r = {}
    rank_c = {}
    rank_rc = {}
    for e in range(ensemble_num):
        for iid in R_dict[e][i].keys():
            if iid in temp_r.keys():
                temp_r[iid].append(R_dict[e][i][iid])
                temp_c[iid].append(C_dict[e][i][iid])
                temp_rc[iid].append(RC_dict[e][i][iid])
            else:
                temp_r[iid] = [R_dict[e][i][iid]]
                temp_c[iid] = [C_dict[e][i][iid]]
                temp_rc[iid] = [RC_dict[e][i][iid]]

        # for rank, (iid-old, scores) in enumerate(sorted(temp_r.items(), key=lambda item: sum(item[1]), reverse=True)):
        #     if iid-old in rank_r.keys():
        #         rank_r[iid-old].append(rank+1)
        #     else:
        #         rank_r[iid-old] = [rank+1]
        # for rank, (iid-old, scores) in enumerate(sorted(temp_c.items(), key=lambda item: sum(item[1]), reverse=True)):
        #     if iid-old in rank_c.keys():
        #         rank_c[iid-old].append(rank + 1)
        #     else:
        #         rank_c[iid-old] = [rank + 1]
        # for rank, (iid-old, scores) in enumerate(sorted(temp_rc.items(), key=lambda item: sum(item[1]), reverse=True)):
        #     if iid-old in rank_rc.keys():
        #         rank_rc[iid-old].append(rank + 1)
        #     else:
        #         rank_rc[iid-old] = [rank + 1]
    for iid in temp_r.keys():
        ER_dict[i][iid] = ensemble_f(temp_r[iid], ensemble_type)
        EC_dict[i][iid] = ensemble_f(temp_c[iid], ensemble_type)
        ERC_dict[i][iid] = ensemble_f(temp_rc[iid], ensemble_type)
        ### recip
        # ER_dict[i][iid-old] = ensemble_f(rank_r[iid-old], 'recip')
        # EC_dict[i][iid-old] = ensemble_f(rank_c[iid-old], 'recip')
        # ERC_dict[i][iid-old] = ensemble_f(rank_rc[iid-old], 'recip')

    sorted_ER = sorted(ER_dict[i].items(), key=lambda item: item[1], reverse=True)
    ER_candidate = [a for a, b in sorted_ER[:k]]
    ER_k.append(ER_candidate)
    ER_corr_arr[i] = gt_dict[i] == np.array(ER_candidate)

    sorted_EC = sorted(EC_dict[i].items(), key=lambda item: item[1], reverse=True)
    EC_candidate = [a for a, b in sorted_EC[:k]]
    EC_k.append(EC_candidate)
    EC_corr_arr[i] = gt_dict[i] == np.array(EC_candidate)

    sorted_RC1 = sorted(RC_dict[0][i].items(), key=lambda item: item[1], reverse=True)
    RC1_candidate = [a for a, b in sorted_RC1[:k]]
    RC1_k.append(RC1_candidate)
    RC1_corr_arr[i] = gt_dict[i] == np.array(RC1_candidate)

    sorted_RC2 = sorted(RC_dict[1][i].items(), key=lambda item: item[1], reverse=True)
    RC2_candidate = [a for a, b in sorted_RC2[:k]]
    RC2_k.append(RC2_candidate)
    RC2_corr_arr[i] = gt_dict[i] == np.array(RC2_candidate)

    sorted_ERC = sorted(ERC_dict[i].items(), key=lambda item: item[1], reverse=True)
    ERC_candidate = [a for a, b in sorted_ERC[:k]]
    ERC_k.append(ERC_candidate)
    ERC_corr_arr[i] = gt_dict[i] == np.array(ERC_candidate)

print(f'hit@{k}-ER: \n{ER_corr_arr.sum() / instance_num}')
print(f'ndcg@{k}-ER: \n{np.sum(ER_corr_arr / np.log2(np.arange(2, k + 2))) / instance_num}')

print(f'hit@{k}-EC: \n{EC_corr_arr.sum() / instance_num}')
print(f'ndcg@{k}-EC: \n{np.sum(EC_corr_arr / np.log2(np.arange(2, k + 2))) / instance_num}')

print(f'hit@{k}-ERC: \n{ERC_corr_arr.sum() / instance_num}')
print(f'ndcg@{k}-ERC: \n{np.sum(ERC_corr_arr / np.log2(np.arange(2, k + 2))) / instance_num}')

print(f'hit@{k}-RC1: \n{RC1_corr_arr.sum() / instance_num}')
print(f'ndcg@{k}-RC1: \n{np.sum(RC1_corr_arr / np.log2(np.arange(2, k + 2))) / instance_num}')

print(f'hit@{k}-RC2: \n{RC2_corr_arr.sum() / instance_num}')
print(f'ndcg@{k}-RC2: \n{np.sum(RC2_corr_arr / np.log2(np.arange(2, k + 2))) / instance_num}')

print('chr')
print((h_all - ER_corr_arr.sum(-1)).sum() / h_all.sum())
print((h_all - EC_corr_arr.sum(-1)).sum() / h_all.sum())
print((h_all - ERC_corr_arr.sum(-1)).sum() / h_all.sum())
