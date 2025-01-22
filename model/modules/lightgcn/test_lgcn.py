import pdb
import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

Recmodel.eval()
Procedure.Test(dataset, Recmodel, 1000)

user_final_embed, item_final_embed = Recmodel.computer()
item_final_embed = item_final_embed.detach().cpu().numpy()

emb_dict = {}

emb_dict['item'] = item_final_embed

import os
import pickle


def save_embedding(root_path, emb_dict):
    with open(os.path.join(root_path, f'emb_ceid.pkl'), 'wb') as fOut:
        pickle.dump(emb_dict, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    return print(f'saving  done!')


# domain = 'Beauty'
# domain = 'Sports_and_Outdoors'
domain = 'Yelp'
save_path = f'../../../data/amazon/filtered/{domain}/'

save_embedding(save_path, emb_dict)
