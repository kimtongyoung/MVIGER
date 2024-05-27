import os
import pdb
import random
from tqdm import tqdm
import numpy as np
import torch
from model.main import T5SequentialRecommender
from model.modules.p5.notebooks.evaluate.metrics4rec import evaluate_all
import pickle
from model.utils import Trie, predict_outputs, prefix_allowed_tokens_fn, save_outputs
from transformers import AutoTokenizer, T5Config


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


from data_loader.amazon_loader import Dset, CrossDset
from torch.utils.data import DataLoader

config = {
    "model_type": "p5",
    "backbone": "t5-small",
    "seed": 2024,
    "act_fn": "relu",
    "lr": 1e-3,
    "code_length": 3,
    "codebook_size": 256,
    # "max_index1": 0,
    # "max_index2": 949,
    # "idx_name1": "sequential_data-Beauty-gid_253.json",
    # "idx_name2": "sequential_data-Beauty-sid_949.json",
    # "max_index1": 0,
    # "max_index2": 1523,
    # "idx_name1": "sequential_data-Sports-gid_302.json",
    # "idx_name2": "sequential_data-Sports-sid_1523.json",
    "max_index1": 0,
    "max_index2": 1364,
    "idx_name1": "sequential_data-Yelp-gid_126.json",
    "idx_name2": "sequential_data-Yelp-sid_1364.json",
    # "index_type": "gid",
    "index_type": "sid",
    # "index_type": "both",
    # "index_type": "cross",
    "test_description_idx": 0,
    "batch_size": 16,
    "beam_size": 20,
    "num_workers": 0,
    "dropout": 0.1,
    "data_dir": "data/amazon/filtered",
    # "domain": "Beauty",
    # "domain": "Sports_and_Outdoors",
    # "domain": "Toys_and_Games",
    "domain": "Yelp",
    "max_length": 512,
    "shuffle": True,
    # "ckpt": "saved/models/Recommender/0517-Beauty-gid-1/model_best_5ep.pth",
    # "ckpt": "saved/models/Recommender/0517-Beauty-gid-2/model_best_5ep.pth",
    # "ckpt": "saved/models/Recommender/0517-Beauty-sid-1/model_best_3ep.pth",
    # "ckpt": "saved/models/Recommender/0517-Beauty-sid-2/model_best_4ep.pth",
    # "ckpt": "saved/models/Recommender/0517-sports-gid-1/model_best_4ep.pth",
    # "ckpt": "saved/models/Recommender/0517-sports-gid-2/model_best_4ep.pth",
    # "ckpt": "saved/models/Recommender/0517-sports-sid-1/model_best_4ep.pth",
    # "ckpt": "saved/models/Recommender/0517-sports-sid-2/model_best_3ep.pth",
    # "ckpt": "saved/models/Recommender/0517-yelp-gid-1/model_best_5ep.pth",
    # "ckpt": "saved/models/Recommender/0517-yelp-gid-2/model_best_4ep.pth",
    # "ckpt": "saved/models/Recommender/0517-yelp-sid-1/model_best_6ep.pth",
    "ckpt": "saved/models/Recommender/0517-yelp-sid-2/model_best_5ep.pth",
    ####
    "use_prefix_trie": True,
}

print(f'start inference : {config["ckpt"]}, temp_num:{config["test_description_idx"]}')

backbone = config['backbone']
root_path = config['data_dir']
t5_config = T5Config.from_pretrained(config['backbone'])
beam_size = config['beam_size']

tokenizer = AutoTokenizer.from_pretrained(config['backbone'])
t5_config.vocab_size = len(tokenizer)
codebook_size = config['codebook_size']
code_length = config['code_length']
new_tokens_g = []
new_tokens_s = []
for code in range(config['codebook_size']):
    for level in range(config['code_length']):
        new_token_g = f'<extra_g_{level}_{code}>'
        new_tokens_g.append(new_token_g)
        new_token_s = f'<extra_s_{level}_{code}>'
        new_tokens_s.append(new_token_s)
for extra_code_g in range(config['max_index1']):
    new_token = f"<extra_g_{config['code_length']}_{extra_code_g}>"
    new_tokens_g.append(new_token)
for extra_code_s in range(config['max_index2']):
    new_token = f"<extra_s_{config['code_length']}_{extra_code_s}>"
    new_tokens_s.append(new_token)
tokenizer.add_tokens(new_tokens_g)
tokenizer.add_tokens(new_tokens_s)
indicator = ['<G>', '<S>']
tokenizer.add_tokens(indicator)

t5_config.vocab_size = len(tokenizer)
model = T5SequentialRecommender(t5_config).from_pretrained(config['backbone'])
model.resize_token_embeddings(t5_config.vocab_size)
print(f'Load from pre-trained model: {config["ckpt"]}')
print(f'total params:{sum(p.numel() for p in model.parameters())}')

from prompt_p5 import task_subgroup_1

infer_set = CrossDset(root_path, config['domain'], 'infer', tokenizer, templates=task_subgroup_1, gid_dict=config['idx_name1'], sid_dict=config['idx_name2'],
                      test_description_idx=config['test_description_idx'], index_type=config['index_type'])

if config['index_type'] == 'gid':
    candidates = infer_set.g_items
elif config['index_type'] == 'sid':
    candidates = infer_set.s_items
else:
    if (config['test_description_idx'] == 0 or config['test_description_idx'] == 2):
        candidates = infer_set.g_items
    else:
        candidates = infer_set.s_items

candidate_trie = Trie([[0] + tokenizer.encode(candidate) for candidate in candidates])
prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

infer_loader = DataLoader(infer_set, shuffle=False, batch_size=config['batch_size'], collate_fn=infer_set.collate_fn)

state_dict = torch.load(config['ckpt'], 'cpu')
model.load_state_dict(state_dict.state_dict())
print('load trained model')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()
print(len(infer_loader))

total_count = 0.
decode_type = 'beam'
with torch.no_grad():
    pred_all = []
    for i, batch in tqdm(enumerate(infer_loader)):
        pred_outs = \
            save_outputs(batch, model, prefix_allowed_tokens, k=beam_size, max_len=20, tokenizer=tokenizer)
        pred_all.append(pred_outs)
        total_count += batch['input_ids'].size(0)

# save test data correction list
corr_all = sum(pred_all, [])

with open(os.path.join('/'.join(config['ckpt'].split('/')[:-1]), f'prediction_list_{beam_size}_{config["test_description_idx"]}'),
          'wb') as fOut:
    pickle.dump(corr_all, fOut, protocol=pickle.HIGHEST_PROTOCOL)
