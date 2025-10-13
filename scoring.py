import json
import numpy as np
from tqdm import tqdm
import pdb


path = 'results_20_20.json'


with open(path, 'r') as f:
    results = json.load(f)


ceid_results = results['ceid']
seid_results = results['seid']



# gt, iid, preds, preds_iid, prior_prob, log_prob, ranks
import math

def hit_at_k(ranks, k, total):
    # ranks: 1-based rank list
    return round(sum(1 for r in ranks if r <= k) / total, 4)

def ndcg_at_k(ranks, k, total):
    # binary relevance = 1 (single relevant item), IDCG@k = 1
    return round(sum((1 / math.log2(r + 1)) if r <= k else 0.0 for r in ranks) / total, 4)

# def scoring_f(x):
#     return np.exp(-x / 10)


# def combine_f(candidates):
#     return 4/5 * candidates[0] + 1/5*candidates[1]


def metric(rank_arr):
    hit5 = sum(rank_arr[:5])
    hit10 = sum(rank_arr[:10])
    dcg_arr = 1 / np.log2(np.arange(2, 10 + 2))
    dcg = rank_arr[:10] * dcg_arr
    ndcg5 = sum(dcg[:5])
    ndcg10 = sum(dcg[:10])
    return hit5, hit10, ndcg5, ndcg10



def mviger_scoring_agg(ceid_results, seid_results, temp=1):
    final_results = {}
    hit5 = 0
    hit10 = 0
    ndcg5 = 0
    ndcg10 = 0
    
    pbar = tqdm(range(len(ceid_results['iid'])))
    for user_idx in pbar:
        final_results[user_idx] = {'ceid_score': {}, 'seid_score': {}, 'all_score': {}, 'final_score': {}, 'sorted_rank': {}}
        # iid = ceid_results['iid'][user_idx]

        c_preds = ceid_results['preds_iid'][user_idx]
        s_preds = seid_results['preds_iid'][user_idx]
        
        c_log_prior = np.array(ceid_results['prior_logit'][user_idx])
        s_log_prior = np.array(seid_results['prior_logit'][user_idx])

        c_prior_probs = np.exp(c_log_prior/temp) / np.sum(np.exp(c_log_prior/temp))
        s_prior_probs = np.exp(s_log_prior/temp) / np.sum(np.exp(s_log_prior/temp))
       
        c_log_probs = ceid_results['log_prob'][user_idx]
        s_log_probs = seid_results['log_prob'][user_idx]

        ll_prob_logits = np.array(ceid_results['log_ll'][user_idx] + seid_results['log_ll'][user_idx])
        ll_prob_probs = np.exp(ll_prob_logits/temp) / np.sum(np.exp(ll_prob_logits/temp))
        c_ll_probs = ll_prob_probs[:10]
        s_ll_probs = ll_prob_probs[10:]
        # pdb.set_trace()

        # pdb.set_trace()
        
        for temp_idx in range(num_temp):
        # for temp_idx in range(10):
            c_pred = c_preds[temp_idx]
            s_pred = s_preds[temp_idx]

            c_prior_prob = c_prior_probs[temp_idx]
            s_prior_prob = s_prior_probs[temp_idx]
            c_ll_prob = c_ll_probs[temp_idx]
            s_ll_prob = s_ll_probs[temp_idx]
                
            c_log_prob = c_log_probs[temp_idx][:b_size]
            s_log_prob = s_log_probs[temp_idx][:b_size]
            # pdb.set_trace()
            
           
            norm_c_prob = norm_score(c_log_prob, tau, scaled) * c_prior_prob
            norm_s_prob = norm_score(s_log_prob, tau, scaled) * s_prior_prob
           
            
            
            # for beam_idx in range(len(c_pred)):
            for beam_idx in range(b_size):
                if c_pred[beam_idx] not in final_results[user_idx]['ceid_score']:
                    final_results[user_idx]['ceid_score'][c_pred[beam_idx]] = []
                final_results[user_idx]['ceid_score'][c_pred[beam_idx]].append(norm_c_prob[beam_idx])

                if c_pred[beam_idx] not in final_results[user_idx]['all_score']:
                    final_results[user_idx]['all_score'][c_pred[beam_idx]] = []
                final_results[user_idx]['all_score'][c_pred[beam_idx]].append(norm_c_prob[beam_idx])
            
                if s_pred[beam_idx] not in final_results[user_idx]['seid_score']:
                    final_results[user_idx]['seid_score'][s_pred[beam_idx]] = []
                final_results[user_idx]['seid_score'][s_pred[beam_idx]].append(norm_s_prob[beam_idx])

                if s_pred[beam_idx] not in final_results[user_idx]['all_score']:
                    final_results[user_idx]['all_score'][s_pred[beam_idx]] = []
                final_results[user_idx]['all_score'][s_pred[beam_idx]].append(norm_s_prob[beam_idx])
    

      
        for iid, score_list in final_results[user_idx]['ceid_score'].items():
            final_results[user_idx]['final_score'][iid] = np.mean(np.array(score_list))

        for iid, score_list in final_results[user_idx]['seid_score'].items():
           
            if iid not in final_results[user_idx]['final_score']:
                final_results[user_idx]['final_score'][iid] = 0
            final_results[user_idx]['final_score'][iid] += np.mean(np.array(score_list))


        sorted_score = sorted(final_results[user_idx]['final_score'].items(), key=lambda item: item[1], reverse=True)
        candidates = [a for a, b in sorted_score[:10]]
        corr_arr = ceid_results['iid'][user_idx] == np.array(candidates)
        
        metric_arr = metric(corr_arr)
        hit5 += metric_arr[0]
        hit10 += metric_arr[1]
        ndcg5 += metric_arr[2]
        ndcg10 += metric_arr[3]
        # pdb.set_trace()
        pbar.set_postfix({f'Hit@5': {hit5 / (user_idx + 1)}, 'Hit@10': {hit10 / (user_idx + 1)}, 'NDCG@5': {ndcg5 / (user_idx + 1)}, 'NDCG@10': {ndcg10 / (user_idx + 1)}})

    return final_results


print(path)
final_results = mviger_scoring_agg(ceid_results, seid_results, temp=1)

