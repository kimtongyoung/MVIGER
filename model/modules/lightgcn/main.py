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
Neg_k = 1


if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")
import os
try:
    best_recall = 0
    ckpt_path = weight_file
    torch.save(Recmodel.state_dict(),ckpt_path)
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[TEST]")
            res = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            if best_recall <= res['recall']:
                os.remove(ckpt_path)
                ckpt_path = weight_file+f'_{epoch}ep'
                torch.save(Recmodel.state_dict(), weight_file+f'_{epoch}ep')
                best_recall = res['recall']
                torch.save(Recmodel.state_dict(), weight_file)
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')
finally:
    if world.tensorboard:
        w.close()


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


user_final_embed, item_final_embed = Recmodel.computer()
final_embed = item_final_embed.detach().cpu().numpy()

emb_dict = {}
emb_dict['item'] = final_embed

import os
import pickle


def save_embedding(root_path, emb_dict):
    with open(os.path.join(root_path, f'emb_ceid.pkl'), 'wb') as fOut:
        pickle.dump(emb_dict, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    return print(f'saving  done!')


save_path = f'../../../data/amazon/filtered/{world.dataset}/'
save_embedding(save_path, emb_dict)
print(f"Collaborative Embedding is saved at : {os.path.join('data/amazon/filtered/', world.dataset, f'emb_ceid.pkl')}")