import pdb

import torch.nn.functional
from torch.utils.data import DataLoader
# import tqdm
from datetime import datetime
from shutil import copyfile
import random
import wandb
from model.rqvae4rec import *


# from model.modules.p5.src.tokenization import P5Tokenizer


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


def create_optimizer_and_scheduler(config, train_loader, model):
    from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
    from torch.optim import AdamW, Adagrad
    steps_per_epoch = len(train_loader)
    t_total = steps_per_epoch * config['epoch']
    warmup_ratio = config['warmup_ratio']
    warmup_iters = int(t_total * warmup_ratio)

    print("steps_per_epoch: %d" % steps_per_epoch)
    print("Total Iters: %d" % t_total)
    print('Warmup ratio:', warmup_ratio)
    print("Warm up Iters: %d" % warmup_iters)

    optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters()], "weight_decay": config['weight_decay']}]
    optim = AdamW(optimizer_grouped_parameters, lr=config['lr'])
    if config['scheduler'] == 'linear':
        lr_scheduler = get_linear_schedule_with_warmup(optim, warmup_iters, t_total)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(optim, warmup_iters)

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

    save_name = config['save_name']

    # wandb logging
    wandb.init(project=config['project_name'], reinit=True)

    now = datetime.now()
    save_dir = f'./saved/models/{config["project_name"]}/{save_name}/'
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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # data_loader
    if config['rqvae_config']['embedding_type'] == 'gid':
        dset = GIDEmbeddingLoader(config['rqvae_config'], user=False)
    elif config['rqvae_config']['embedding_type'] == 'uid':
        dset = GIDEmbeddingLoader(config['rqvae_config'], user=True)
    elif config['rqvae_config']['embedding_type'] == 'sid':
        dset = SemIDEmbeddingLoader(config['rqvae_config'])
    elif config['rqvae_config']['embedding_type'] == 'hid':
        dset = HIDEmbeddingLoader(config['rqvae_config'], config['rqvae_config']['mix_type'])
    loader = DataLoader(dset, shuffle=config['shuffle'], batch_size=config['batch_size'],
                        collate_fn=dset.collate_fn)
    optim, lr_scheduler = create_optimizer_and_scheduler(config, loader, model)

    ckpt_path = os.path.join(save_dir, f'model_best_1ep.pth')
    if config['distributed']:
        dist.barrier()

    curr_best = 10000.
    for epoch in range(config['epoch']):
        # train
        model.train()
        train_loss = 0.
        recon_loss = 0.
        latent_loss = 0.
        for step_t, batch in enumerate(loader):
            # pdb.set_trace()
            inp = batch['input_emb'].to(device)
            outputs = model(inp)
            # out, quant_loss, split-gcn-code
            outputs = model.compute_loss(*outputs, x=inp)
            loss = outputs['loss_total']
            loss_reconstruct = outputs['loss_recon']
            loss_latent = outputs['loss_latent']
            train_loss += loss
            recon_loss += loss_reconstruct
            latent_loss += loss_latent
            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_scheduler.step()
            lr = lr_scheduler.get_last_lr()[0]

        model.eval()
        code_outs = []
        for step_v, batch in enumerate(loader):
            # pdb.set_trace()
            inp = batch['input_emb'].to(device)
            outputs = model(inp)
            # out, quant_loss, split-gcn-code
            codes = outputs[2]
            code_outs.append(codes)
        total_code_tensor = torch.cat(code_outs, 0)
        collision_num, calc_dict = calc_collision_num(total_code_tensor)
        code_length = config['rqvae_config']['code_len']
        codebook_size = config['rqvae_config']['n_embed']
        lv_dict = generate_dict(codebook_size, code_length)
        lv_dict = count_dict(total_code_tensor, lv_dict, code_length)

        std_list, count = calc_div(lv_dict, code_length, codebook_size)
        std = np.mean(std_list)
        per = calc_percent(count)
        # pdb.set_trace()
        #
        print(f'{epoch + 1} train loss: {train_loss / step_t}, recon: {recon_loss / step_t},'
              f' latent: {latent_loss / step_t} last_lr: {lr}')
        print(f' std: {std}, std_list: {std_list}')
        print(f' collision_num: {collision_num}, lv usage: {per}')
        print('###############################################################')
        wandb.log({"train_loss": train_loss / step_t, 'epoch': epoch})
        wandb.log({"recon_loss": recon_loss / step_t, 'epoch': epoch})
        wandb.log({"latent_loss": latent_loss / step_t, 'epoch': epoch})
        wandb.log({"collision_num": collision_num, 'epoch': epoch})
        for length in range(code_length):
            wandb.log({f"lv{length + 1} usage": per[length], 'epoch': epoch})
        wandb.log({"lr": lr, 'epoch': epoch})

        if per[0] >= 80:
            if collision_num <= curr_best:
                if os.path.isfile(ckpt_path):
                    os.remove(ckpt_path)
                curr_best = collision_num
                torch.save(model.state_dict(), f'{save_dir}model_best_{epoch + 1}ep.pth')
                ckpt_path = os.path.join(save_dir, f'model_best_{epoch + 1}ep.pth')
        if (epoch + 1) % 1000 == 0:
            torch.save(model.state_dict(), f'{save_dir}model_{epoch + 1}ep.pth')
