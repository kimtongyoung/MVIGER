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
from data_loader.prompt_p5 import task_subgroup_1
from data_loader.mviger_loader import MVIGERDset
from torch.utils.data import DataLoader
from collections import defaultdict
# from scoring import mviger_scoring_agg
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)




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



def inference(config):
    print(f'start inference : {config["ckpt"]}')
    root_path = config['data_dir']
    t5_config = T5Config.from_pretrained('t5-small')
    max_index1 = int(config['idx_name1'].split('_')[1])
    max_index2 = int(config['idx_name2'].split('_')[1])
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    tokenizer_prior = AutoTokenizer.from_pretrained('t5-small')

    if config['use_inst']:
        new_tokens_c = ['<CEID>']
        new_tokens_s = ['<SEID>']
        new_tokens_i = ['<IID>']
    else:   
        new_tokens_c = []
        new_tokens_s = []
        new_tokens_i = []
    if config['pooling_type'] == 'sos':
        new_tokens_i.append('<Prior>')

    # CEID and SEID
    for code in range(config['codebook_size']):
        for level in range(config['code_length']):
            new_token_c = f'<CEID_{level}_{code}>'
            new_tokens_c.append(new_token_c)
            new_token_s = f'<SEID_{level}_{code}>'
            new_tokens_s.append(new_token_s)
    # leaf nodes
    max_index1 = int(config['idx_name1'].split('_')[1])
    max_index2 = int(config['idx_name2'].split('_')[1])

    for extra_code_c in range(max_index1):
        new_token = f"<CEID_{config['code_length']}_{extra_code_c}>"
        new_tokens_c.append(new_token)
    for extra_code_s in range(max_index2):
        new_token = f"<SEID_{config['code_length']}_{extra_code_s}>"
        new_tokens_s.append(new_token)            
    tokenizer.add_tokens(new_tokens_c)
    tokenizer.add_tokens(new_tokens_s)

    for iid in range(config['num_items']):
        new_token = f'<IID_{iid}>'
        new_tokens_i.append(new_token)       
    tokenizer_prior.add_tokens(new_tokens_i)


    t5_config = T5Config.from_pretrained('t5-small', update_kwargs={'vocab_size_prior': len(tokenizer_prior)})

    t5_config.vocab_size = len(tokenizer)
    t5_config.update({'vocab_size_prior': len(tokenizer_prior)})

    model = T5SequentialRecommender.from_pretrained('t5-small', config=t5_config, ignore_mismatched_sizes=True)

    state_dict = torch.load(config['ckpt'], 'cpu').state_dict()


    model.load_state_dict(state_dict, strict=False)
    test_set = MVIGERDset(root_path, config['domain'], 'test', tokenizer, tokenizer_prior,  prompt_templates=task_subgroup_1, ceid_dict=config['idx_name1'], seid_dict=config['idx_name2'],
                         num_templates=config['num_templates'], num_indexes=config['num_indexes'],  use_inst=config['use_inst'])
    test_loader = DataLoader(test_set, batch_size=config['test_batch_size'], collate_fn=test_set.collate_fn, num_workers=config['num_workers'])


    
    candidates_c = test_set.c_items
    candidate_trie_c = Trie([[0] + tokenizer.encode(candidate) for candidate in candidates_c])
    prefix_allowed_tokens_c = prefix_allowed_tokens_fn(candidate_trie_c)

    candidates_s = test_set.s_items
    candidate_trie_s = Trie([[0] + tokenizer.encode(candidate) for candidate in candidates_s])
    prefix_allowed_tokens_s = prefix_allowed_tokens_fn(candidate_trie_s)
    
    constraints = [prefix_allowed_tokens_c, prefix_allowed_tokens_s]
  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    idx_iid = test_set.idx_iid

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="VIGER inference: ")
        total_hit5 = 0.
        total_hit10 = 0.
        total_ndcg5 = 0.
        total_ndcg10 = 0.
        total_instance = 0.
      
        all_results = {
            'ceid': {
                'gt': [],
                'iid': [],
                'preds' : [],
                'preds_iid':[],
                'prior_prob' : [],
                "prior_logit": [],
                'log_prob' : [],
                'log_ll': [],
                'ranks' : [],
            }, 
            'seid': {
                'gt': [],
                'iid': [],
                'preds' : [],
                'preds_iid':[],
                'prior_prob' : [],
                "prior_logit": [],
                'log_prob' : [],
                'log_ll': [],
                'ranks' : [],
            }
        }



        for i, batch in enumerate(pbar):
            total_instance += batch['z']['input_ids'].size(0)//(config['num_indexes']*config['num_templates'])
            results = model.inference_all(batch, k=config['beam_size'], max_len=20, constraints=constraints, tokenizer=tokenizer, idx_iid=idx_iid, index_list=config['index_list'], template_num=config['test_template_num'], pooling_type=config['pooling_type'])

            
            ceid_results = results['ceid']
            seid_results = results['seid']

            for key,value in ceid_results.items():
                all_results['ceid'][key].extend(value)
            for key,value in seid_results.items():
                all_results['seid'][key].extend(value)

        ep = config['ckpt'].split('/')[-1].split('_')[-1].split('ep')[0]
        with open(os.path.join('/'.join(config['ckpt'].split('/')[:-1]), f"results_{ep}_{config['beam_size']}.json"), "w", encoding="utf-8") as file:
            json.dump(all_results, file, indent=4)

   