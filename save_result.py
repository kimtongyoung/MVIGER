import os
import pdb
import random
from tqdm import tqdm
import numpy as np
import torch
from model.main import T5SequentialRecommender
import pickle
from model.utils import Trie, predict_outputs, prefix_allowed_tokens_fn, save_outputs
from transformers import AutoTokenizer, T5Config
from prompt_p5 import task_subgroup_1 as p5_prompt


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


import json


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def ReadLineFromFile(path):
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines


def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    # Change Multi GPU to single GPU
    original_keys = list(state_dict.state_dict().keys())
    for key in original_keys:
        if key.startswith("module."):
            new_key = key[len("module."):]
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def load_checkpoint(model, ckpt_path):
    state_dict = load_state_dict(ckpt_path, 'cpu')
    results = model.load_state_dict(state_dict, strict=False)
    print('Model loaded from ', ckpt_path)


from data_loader.amazon_loader import SCRecDset
from torch.utils.data import DataLoader
from collections import defaultdict


def scoring_f(x):
    return np.exp(-x / 10)


def combine_f(candidates):
    return 4/5 * candidates[0] + 1/5*candidates[1]


def aggrergation_f(candidates):
    return np.sum(candidates)


def metric(rank_arr):
    hit5 = sum(rank_arr[:5])
    hit10 = sum(rank_arr[:10])
    dcg_arr = 1 / np.log2(np.arange(2, 10 + 2))
    dcg = rank_arr[:10] * dcg_arr
    ndcg5 = sum(dcg[:5])
    ndcg10 = sum(dcg[:10])
    return hit5, hit10, ndcg5, ndcg10


def reranking(pred_c, pred_s, idx_iid):
    gt = idx_iid[pred_c[0][0]]
    rank_c = {}
    rank_s = {}
    freq_c = {}
    freq_s = {}
    score = {}
    for i, pred_i in enumerate(pred_c):
        for rank, pred in enumerate(pred_i[1]):
            if idx_iid[pred] in freq_c.keys():
                freq_c[idx_iid[pred]] += 1
                rank_c[idx_iid[pred]].append(rank)
            else:
                freq_c[idx_iid[pred]] = 1
                rank_c[idx_iid[pred]] = [rank]
    for i, pred_i in enumerate(pred_s):
        for rank, pred in enumerate(pred_i[1]):
            if idx_iid[pred] in freq_s.keys():
                freq_s[idx_iid[pred]] += 1
                rank_s[idx_iid[pred]].append(rank)
            else:
                freq_s[idx_iid[pred]] = 1
                rank_s[idx_iid[pred]] = [rank]

    for iid, rank_list in rank_c.items():
        mean_rank = np.mean(rank_list)
        position_score = scoring_f(mean_rank)
        n = len(rank_list)
        if n < 2:
            var_x = 20 ** 2
        else:
            var_x = np.sum((np.array(rank_list) - mean_rank) ** 2) / (n - 1)
        sstd = np.sqrt(var_x)
        consistency_score = scoring_f(sstd)
        score[iid] = combine_f([position_score, consistency_score])

    for iid, rank_list in rank_s.items():
        mean_rank = np.mean(rank_list)
        position_score = scoring_f(mean_rank)
        n = len(rank_list)
        if n < 2:
            var_x = 20 ** 2
        else:
            var_x = np.sum((np.array(rank_list) - mean_rank) ** 2) / (n - 1)
        sstd = np.sqrt(var_x)
        consistency_score = scoring_f(sstd)
        if iid in score.keys():
            score[iid] += combine_f([position_score, consistency_score])
        else:
            score[iid] = combine_f([position_score, consistency_score])

    sorted_score = sorted(score.items(), key=lambda item: item[1], reverse=True)
    candidates = [a for a, b in sorted_score[:10]]
    corr_arr = gt == np.array(candidates)
    return corr_arr


def inference(config):
    print(f'start inference : {config["ckpt"]}')
    root_path = config['data_dir']
    t5_config = T5Config.from_pretrained('t5-small')
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    new_tokens_c = []
    new_tokens_s = []

    if config['is_p5id']:
        with open(os.path.join(config['data_dir'], config['domain'], config['idx_name1'])) as f:
            p5cid_dict = json.load(f)
        with open(os.path.join(config['data_dir'], config['domain'], config['idx_name2'])) as ff:
            p5semid_dict = json.load(ff)

        for code in p5cid_dict[1]:
            new_token_c = f'<extra_c_{code}>'
            new_tokens_c.append(new_token_c)
        for code in p5semid_dict[1]:
            new_token_s = f'<extra_s_{code}>'
            new_tokens_s.append(new_token_s)
        tokenizer.add_tokens(new_tokens_c)
        tokenizer.add_tokens(new_tokens_s)
        indicator = ['<C>', '<S>']
        tokenizer.add_tokens(indicator)
    else:
        max_index1 = int(config['idx_name1'].split('_')[1])
        max_index2 = int(config['idx_name2'].split('_')[1])

        for code in range(config['codebook_size']):
            for level in range(config['code_length']):
                new_token_c = f'<extra_c_{level}_{code}>'
                new_tokens_c.append(new_token_c)
                new_token_s = f'<extra_s_{level}_{code}>'
                new_tokens_s.append(new_token_s)
        for extra_code_c in range(max_index1):
            new_token = f"<extra_c_{config['code_length']}_Leaf{extra_code_c}>"
            new_tokens_c.append(new_token)
        for extra_code_s in range(max_index2):
            new_token = f"<extra_s_{config['code_length']}_Leaf{extra_code_s}>"
            new_tokens_s.append(new_token)
        tokenizer.add_tokens(new_tokens_c)
        tokenizer.add_tokens(new_tokens_s)
        indicator = ['<C>', '<S>']
        tokenizer.add_tokens(indicator)

    t5_config.vocab_size = len(tokenizer)
    model = T5SequentialRecommender(t5_config).from_pretrained('t5-small')
    model.resize_token_embeddings(t5_config.vocab_size)
    print(f'Load from pre-trained model: {config["ckpt"]}')
    print(f'total params:{sum(p.numel() for p in model.parameters())}')


    prompt = p5_prompt


    infer_set_c = SCRecDset(root_path, config['domain'], 'infer', tokenizer, templates=prompt, ceid_dict=config['idx_name1'], seid_dict=config['idx_name2'],
                            num_templates=config['num_templates'], test_instruction_type='ceid', is_p5id=config['is_p5id'])
    infer_set_s = SCRecDset(root_path, config['domain'], 'infer', tokenizer, templates=prompt, ceid_dict=config['idx_name1'], seid_dict=config['idx_name2'],
                            num_templates=config['num_templates'], test_instruction_type='seid', is_p5id=config['is_p5id'])
    candidates_c = infer_set_c.c_items
    candidate_trie_c = Trie([[0] + tokenizer.encode(candidate) for candidate in candidates_c])
    prefix_allowed_tokens_c = prefix_allowed_tokens_fn(candidate_trie_c)
    infer_loader_c = DataLoader(infer_set_c, shuffle=False, batch_size=config['num_templates'], collate_fn=infer_set_c.collate_fn)

    candidates_s = infer_set_s.s_items
    candidate_trie_s = Trie([[0] + tokenizer.encode(candidate) for candidate in candidates_s])
    prefix_allowed_tokens_s = prefix_allowed_tokens_fn(candidate_trie_s)
    infer_loader_s = DataLoader(infer_set_s, shuffle=False, batch_size=config['num_templates'], collate_fn=infer_set_s.collate_fn)

    state_dict = torch.load(config['ckpt'], 'cpu')
    model.load_state_dict(state_dict.state_dict())
    print('load trained model')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    print(len(infer_loader_c))

    iid_gid = infer_set_c.iid_ceid
    iid_sid = infer_set_c.iid_seid
    gid_iid = infer_set_c.ceid_iid
    sid_iid = infer_set_c.seid_iid
    idx_iid = infer_set_c.idx_iid

    import time
    result_c = []
    result_s = []
    with torch.no_grad():
        pbar = tqdm(zip(infer_loader_c, infer_loader_s), desc="SC-Rec inference: ")
        total_hit5 = 0.
        total_hit10 = 0.
        total_ndcg5 = 0.
        total_ndcg10 = 0.
        for i, batch in enumerate(pbar):
            batch1, batch2 = batch
            pred_outs_c = save_outputs(batch1, model, prefix_allowed_tokens_c, k=config['beam'], max_len=30, tokenizer=tokenizer)
            pred_outs_s = save_outputs(batch2, model, prefix_allowed_tokens_s, k=config['beam'], max_len=30, tokenizer=tokenizer)

            result_c.append(pred_outs_c)
            result_s.append(pred_outs_s)
            final_rank = reranking(pred_outs_c, pred_outs_s, idx_iid)

            hit5, hit10, ndcg5, ndcg10 = metric(final_rank)
            total_hit5 += hit5
            total_hit10 += hit10
            total_ndcg5 += ndcg5
            total_ndcg10 += ndcg10
            pbar.set_postfix({f'Hit@5': {total_hit5 / (i + 1)}, 'Hit@10': {total_hit10 / (i + 1)}, 'NDCG@5': {total_ndcg5 / (i + 1)}, 'NDCG@10': {total_ndcg10 / (i + 1)}})
        print(f'Hit@5 : {total_hit5 / (i + 1)}, Hit@10 : {total_hit10 / (i + 1)}, NDCG@5 : {total_ndcg5 / (i + 1)}, NDCG@10 : {total_ndcg10 / (i + 1)}')
        print('###############################################################')

    with open(os.path.join('/'.join(config['ckpt'].split('/')[:-1]), "result_c.json"), "w", encoding="utf-8") as file:
        json.dump(result_c, file, indent=4)
    with open(os.path.join('/'.join(config['ckpt'].split('/')[:-1]), "result_s.json"), "w", encoding="utf-8") as file:
        json.dump(result_s, file, indent=4)