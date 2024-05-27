import os
import pdb

from tqdm import tqdm
import numpy as np
import torch
from model.rqvae4rec import RQVAE, SemIDEmbeddingLoader, GIDEmbeddingLoader, HIDEmbeddingLoader
import pickle


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


import json


def save_json(dict, path):
    json_str = json.dumps(dict)
    with open(path, 'w') as out:
        out.write(json_str)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def ReadLineFromFile(path):
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines


def create_model(model_class, backbone, t5_config=None):
    model = model_class.from_pretrained(
        backbone,
        config=t5_config
    )
    return model


def create_config(config):
    from transformers import T5Config
    config_class = T5Config
    t5_config = config_class.from_pretrained(config['backbone'])
    t5_config.dropout_rate = config['dropout']
    t5_config.dropout = config['dropout']
    t5_config.attention_dropout = config['dropout']
    t5_config.activation_dropout = config['dropout']

    return t5_config


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


def generate_dict(codebook_size, code_length):
    lv_dict = {}
    for length in range(code_length):
        lv_dict[length + 1] = defaultdict(int)
        for code in range(codebook_size[length]):
            lv_dict[length + 1][code] = 0
    return lv_dict


def count_dict(codes, lv_dict, code_length):
    for code in codes:
        for length in range(code_length):
            lv_dict[length + 1][int(code[length])] += 1
    return lv_dict


def calc_div(lv_dict, code_length, codebook_size):
    dev = []
    for length in range(code_length):
        temp = []
        for i in range(codebook_size[length]):
            temp.append(lv_dict[length + 1][i])
        dev.append(temp)

    return [np.std(dev[i]) for i in range(len(dev))], dev


def calc_percent(dev):
    return [(np.array(dev[i]) != 0).sum() / len(dev[i]) * 100 for i in range(len(dev))]


from torch.utils.data import DataLoader

config = {
    "embedding_type": "gid",
    # "embedding_type": "sid",
    # "embedding_type": "uid",
    # "embedding_type": "hid",
    # "backbone": "t5-small",
    "max_length": 512,
    "batch_size": 4096,
    "norm_type": 'no',
    # "ckpt": "saved/models/rq-vae/Toys-sid-768-batch/model_10000ep.pth",
    # "idx_name": "Toys-sid-768-batch",
    # "ckpt": "saved/models/rq-vae/Toys-gid-best-batch/model_10000ep.pth",
    # "idx_name": "Toys-gid-best-batch",
    # "ckpt": "saved/models/rq-vae/Beauty-gid-best-batch/model_10000ep.pth",
    # "idx_name": "Beauty-gid-best-batch",
    # "ckpt": "saved/models/rq-vae/Beauty-sid-768-batch/model_10000ep.pth",
    # "idx_name": "Beauty-sid-768-batch",
    "ckpt": "saved/models/rq-vae/Yelp-gid-best-batch-nodecay/model_10000ep.pth",
    "idx_name": "Yelp-gid-best-batch-nodecay",
    # "ckpt": "saved/models/rq-vae/Yelp-sid-768-batch-nodecay/model_10000ep.pth",
    # "idx_name": "Yelp-sid-768-batch-nodecay",
    # "ckpt": "saved/models/rq-vae/Sports-gid-best-batch/model_10000ep.pth",
    # "idx_name": "Sports-gid-best-batch",
    # "ckpt": "saved/models/rq-vae/Sports-sid-768-batch/model_10000ep.pth",
    # "idx_name": "Sports-sid-768-batch",
    "gid_name": "embedding-768-best",
    "sid_name": "t5-base",
    "data_dir": "data/amazon/filtered",
    "domain": "Yelp",
    "k_iter": 100,
    # "mix_type": "avg",
    "n_embed": [256, 256, 256],
    "embed_dim": 32,
    "code_len": 3,
    "n_layers": 4,
    "dims": [
        768,
        512,
        256,
        128,
        64,
        32
    ],
    "beta": 0.25,
    "act": "relu",
    "shared_codebook": False,
    "restart_unused_codes": True,
    "drop": 0
}

idx_name = config['idx_name']
# def test(config):
root_path = config['data_dir']

model = RQVAE(config)
print(f'trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
if config['embedding_type'] == 'gid':
    test_set = GIDEmbeddingLoader(config)
elif config['embedding_type'] == 'uid':
    test_set = GIDEmbeddingLoader(config, user=True)
elif config['embedding_type'] == 'sid':
    test_set = SemIDEmbeddingLoader(config)
elif config['embedding_type'] == 'hid':
    test_set = HIDEmbeddingLoader(config, config['mix_type'])
test_loader = DataLoader(test_set, batch_size=config['batch_size'], collate_fn=test_set.collate_fn)

state_dict = torch.load(config['ckpt'], 'cpu')
model.load_state_dict(state_dict)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()
print(len(test_loader))

from collections import defaultdict

item_idx_list = []
code_outs = []
total_count = 0
with torch.no_grad():
    for i, batch in tqdm(enumerate(test_loader)):
        inp = batch['input_emb'].to(device)
        codes = model.get_codes(inp)
        item_idx = batch['item_idx']
        item_idx_list.append(item_idx)
        code_outs.append(codes)
        batch_size = inp.size(0)
        total_count += batch['input_emb'].size(0)

total_code_tensor = torch.cat(code_outs, 0)

codebook_size = config['n_embed']
code_length = config['code_len']
lv_dict = generate_dict(codebook_size, code_length)
lv_dict = count_dict(total_code_tensor, lv_dict, code_length)
std_list, count = calc_div(lv_dict, code_length, codebook_size)
std = np.mean(std_list)
per = calc_percent(count)

print(f' std_total: {std}, std_list: {std_list}')
print(f' lv usage: {per}')
print('###############################################################')


def create_code_dict(item_idx_list, total_code_tensor):
    code_dict = {}
    arr = np.concatenate(item_idx_list, 0).tolist()
    for idx, item_idx in enumerate(arr):
        code_dict[item_idx] = total_code_tensor[idx].tolist()
    return code_dict


result = create_code_dict(item_idx_list, total_code_tensor)


def collision_correcting(result):
    res_out = {}
    token_size = config['n_embed']
    id_count = {}
    collision_idx = 1
    for iid, code in result.items():
        raw_id = ','.join([str(c) for c in code])
        if raw_id not in id_count:
            id_count[raw_id] = 1
            idx = raw_id + ',0'
            idx = idx.split(',')
            res_out[iid] = np.array(idx, dtype=int).tolist()
        else:
            id_count[raw_id] += 1
            idx = raw_id + f',{collision_idx}'
            collision_idx += 1
            idx = idx.split(',')
            res_out[iid] = np.array(idx, dtype=int).tolist()
    print(collision_idx)
    return res_out, collision_idx


# def collision_handling_012(result):
#     res_out = {}
#     token_size = config['n_embed']
#     id_count = {}
#     max_collision = 0
#     for iid-old, code in result.items():
#         raw_id = ','.join([str(c) for c in code])
#         if raw_id not in id_count:
#             id_count[raw_id] = 1
#             idx = raw_id + f',{0}'
#             idx = idx.split(',')
#             res_out[iid-old] = np.array(idx, dtype=int).tolist()
#         else:
#             idx = raw_id + f',{id_count[raw_id]}'
#             id_count[raw_id] += 1
#             idx = idx.split(',')
#             res_out[iid-old] = np.array(idx, dtype=int).tolist()
#     for k, v in id_count.items():
#         if v >= max_collision:
#             max_collision = v
#     print(max_collision)
#     return res_out, max_collision


res_out, collision_idx = collision_correcting(result)
# res_out, collision_idx = collision_handling_012(result)

path = f'data/amazon/filtered/{config["domain"]}/rqid/rq-{config["embedding_type"]}/sequential_data_{idx_name}_{collision_idx}.json'
save_json(res_out, path)
