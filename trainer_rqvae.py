import json
import random
from collections import defaultdict
from shutil import copyfile
import wandb
import torch.nn.functional
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from model.rqvae4rec import *
from datetime import datetime


def save_json(dict, path):
    json_str = json.dumps(dict)
    with open(path, 'w') as out:
        out.write(json_str)


def generate_dict(codebook_size, code_length):
    lv_dict = {}
    for length in range(code_length):
        lv_dict[length + 1] = defaultdict(int)
        for code in range(codebook_size):
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
        for i in range(codebook_size):
            temp.append(lv_dict[length + 1][i])
        dev.append(temp)

    return [np.std(dev[i]) for i in range(len(dev))], dev


def calc_percent(dev):
    return [(np.array(dev[i]) != 0).sum() / len(dev[i]) * 100 for i in range(len(dev))]


def create_optimizer_and_scheduler(config, train_loader, model):
    steps_per_epoch = len(train_loader)
    t_total = steps_per_epoch * config['epoch']
    warmup_ratio = config['warmup_ratio']
    warmup_iters = int(t_total * warmup_ratio)

    print("steps_per_epoch: %d" % steps_per_epoch)
    print("Total Iters: %d" % t_total)
    print('Warmup ratio:', warmup_ratio)
    print("Warm up Iters: %d" % warmup_iters)

    optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters()]}]
    optim = AdamW(optimizer_grouped_parameters, lr=config['lr'], weight_decay=0)
    lr_scheduler = get_linear_schedule_with_warmup(optim, warmup_iters, t_total)

    return optim, lr_scheduler


def calc_collision_num(total_code_tensor):
    calc_dict = {}
    collision_num = 0
    for code in total_code_tensor:
        code = tuple(code.tolist())
        if code in calc_dict.keys():
            calc_dict[code] += 1
            collision_num += 1
        else:
            calc_dict[code] = 1
    return collision_num, calc_dict


def rqvae_runner(config):
    # set rid seed
    SEED = config['seed']
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    save_name = config['domain']+'-'+config['save_name']
    save_dir = f'./saved/models/{config["project_name"]}/{save_name}/'
    ############## wandb logging
    run = wandb.init(project=config['project_name'], reinit=True)
    now = datetime.now()
    name = save_name + now.strftime('-%Y-%m-%d-%H%M%S')
    wandb.run.name = name
    wandb.run.save()

    if os.path.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)

    os.makedirs(save_dir)
    copyfile(config['config_name'], save_dir + config['config_name'])

    model = RQVAE(config['rqvae_config'])
    print(f'trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'training {save_name}')


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # data_loader
    if config['rqvae_config']['embedding_type'] == 'ceid':
        dset = CEIDEmbeddingLoader(config['rqvae_config'])

    else:
        dset = SEIDEmbeddingLoader(config['rqvae_config'])

    loader = DataLoader(dset, shuffle=True, batch_size=config['batch_size'], collate_fn=dset.collate_fn)
    optim, lr_scheduler = create_optimizer_and_scheduler(config, loader, model)

    for epoch in tqdm(range(config['epoch'])):
        # train
        model.train()
        train_loss = 0.
        recon_loss = 0.
        latent_loss = 0.
        for step_t, batch in enumerate(loader, start=1):
            inp = batch['input_emb'].to(device)
            outputs = model(inp)
            outputs = model.compute_loss(*outputs, x=inp)
            loss = outputs['loss_total']
            loss_reconstruct = outputs['loss_recon']
            loss_latent = outputs['loss_latent']
            train_loss += loss
            recon_loss += loss_reconstruct
            latent_loss += loss_latent
            optim.zero_grad()
            loss.backward()
            if config['grad_clip'] != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optim.step()
            lr_scheduler.step()
            lr = lr_scheduler.get_last_lr()[0]

        model.eval()
        code_outs = []
        with torch.no_grad():
            for step_v, batch in enumerate(loader):
                inp = batch['input_emb'].to(device)
                codes = model.get_codes(inp)
                code_outs.append(codes)
        total_code_tensor = torch.cat(code_outs, 0)

        collision_num, calc_dict = calc_collision_num(total_code_tensor)
        code_length = config['code_length']
        codebook_size = config['codebook_size']
        lv_dict = generate_dict(codebook_size, code_length)
        lv_dict = count_dict(total_code_tensor, lv_dict, code_length)

        std_list, count = calc_div(lv_dict, code_length, codebook_size)
        std = np.mean(std_list)
        per = calc_percent(count)
        # pdb.set_trace()

        print(f'{epoch + 1} train loss: {train_loss / step_t}, recon: {recon_loss / step_t}, latent: {latent_loss / step_t} last_lr: {lr}')
        print(f' std: {std}, std_list: {std_list}')
        print(f' collision_num: {collision_num}, lv usage: {per}')
        print('###############################################################')
        wandb.log({"train_loss": train_loss / step_t, 'epoch': epoch})
        wandb.log({"recon_loss": recon_loss / step_t, 'epoch': epoch})
        wandb.log({"latent_loss": latent_loss / step_t, 'epoch': epoch})
        wandb.log({"collision_num": collision_num, 'epoch': epoch})
        wandb.log({"lr": lr, 'epoch': epoch})
        for length in range(code_length):
            wandb.log({f"lv{length + 1} usage": per[length], 'epoch': epoch})
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f'{save_dir}model_{epoch + 1}ep.pth')

    run.finish()

    model = RQVAE(config['rqvae_config'])
    state_dict = torch.load(f'{save_dir}model_{config["epoch"]}ep.pth', 'cpu')
    model.load_state_dict(state_dict)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    item_idx_list = []
    code_outs = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            inp = batch['input_emb'].to(device)
            codes = model.get_codes(inp)
            item_idx = batch['item_idx']
            item_idx_list.append(item_idx)
            code_outs.append(codes)

    total_code_tensor = torch.cat(code_outs, 0)

    codebook_size = config['codebook_size']
    code_length = config['code_length']
    lv_dict = generate_dict(codebook_size, code_length)
    lv_dict = count_dict(total_code_tensor, lv_dict, code_length)

    def create_code_dict(item_idx_list, total_code_tensor):
        code_dict = {}
        arr = np.concatenate(item_idx_list, 0).tolist()
        for idx, item_idx in enumerate(arr):
            code_dict[item_idx] = total_code_tensor[idx].tolist()
        return code_dict

    result = create_code_dict(item_idx_list, total_code_tensor)

    def collision_handling(result):
        res_out = {}
        token_size = config['codebook_size']
        id_count = {}
        collision_idx = 1
        for iid, code in result.items():
            raw_id = ','.join([str(c) for c in code])
            if raw_id not in id_count:
                id_count[raw_id] = 1
                idx = raw_id + ',Leaf0'
                idx = idx.split(',')
                res_out[iid] = idx
            else:
                id_count[raw_id] += 1
                idx = raw_id + f',Leaf{collision_idx}'
                collision_idx += 1
                idx = idx.split(',')
                res_out[iid] = idx
        print(collision_idx)
        return res_out, collision_idx

    res_out, collision_idx = collision_handling(result)

    path = f'data/amazon/filtered/{config["domain"]}/sequential-data-{save_name}_{collision_idx}_.json'
    save_json(res_out, path)

    print(f'ID is created at : {path}, with extra {collision_idx}  tokens')
