import pdb
import torch.nn.functional
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime
import os
from shutil import copyfile
from model.main import *
import random
import wandb
from model.utils import *
from transformers import AutoTokenizer
from data_loader.amazon_loader import Dset, CrossDset
from prompt_p5 import task_subgroup_1

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "100"


#
# def create_config(config):
#     from transformers import T5Config
#     config_class = T5Config
#     t5_config = config_class.from_pretrained(config['backbone'])
#     t5_config.dropout_rate = config['dropout']
#     t5_config.dropout = config['dropout']
#     t5_config.dense_act_fn = config['act_fn']
#     t5_config.attention_dropout = config['dropout']
#     t5_config.activation_dropout = config['dropout']
#     t5_config.d_ff = config['T5Config']['d_ff']
#     t5_config.d_kv = config['T5Config']['d_kv']
#     t5_config.d_model = config['T5Config']['d_model']
#     t5_config.num_decoder_layers = config['T5Config']['num_decoder_layers']
#     t5_config.num_heads = config['T5Config']['num_heads']
#     t5_config.num_layers = config['T5Config']['num_layers']
#     return t5_config


def create_model(model_class, backbone, t5_config=None):
    model = model_class.from_pretrained(backbone, config=t5_config)
    return model


def create_optimizer_and_scheduler(config, train_loader, model):
    from transformers.optimization import get_linear_schedule_with_warmup
    from torch.optim import AdamW
    steps_per_epoch = len(train_loader)
    t_total = steps_per_epoch * config['epoch']
    warmup_ratio = config['warmup_ratio']
    warmup_iters = int(t_total * warmup_ratio)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": config['weight_decay']},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    optim = AdamW(optimizer_grouped_parameters, lr=config['lr'], )
    lr_scheduler = get_linear_schedule_with_warmup(optim, warmup_iters, t_total)
    print("steps_per_epoch: %d" % steps_per_epoch)
    print("Total Iters: %d" % t_total)
    print('Warmup ratio:', warmup_ratio)
    print("Warm up Iters: %d" % warmup_iters)
    return optim, lr_scheduler


def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    original_keys = list(state_dict.keys())
    for key in original_keys:
        if key.startswith("module."):
            new_key = key[len("module."):]
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def p5_runner(config):
    SEED = config['seed']
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    save_name = config['save_name']
    # wandb logging
    run = wandb.init(project=config['project_name'], reinit=True)
    now = datetime.now()
    save_dir = f'./saved/models/{config["project_name"]}/{save_name}/'
    name = save_name + now.strftime('-%Y-%m-%d-%H%M%S')
    wandb.run.name = name
    wandb.run.save()
    # initialize save path
    if os.path.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    copyfile(config['config_name'], save_dir + config['config_name'])
    root_path = config['data_dir']
    # prepare model, tokenizer for train
    from transformers import T5Config
    t5_config = T5Config.from_pretrained(config['backbone'])
    tokenizer = AutoTokenizer.from_pretrained(config['backbone'])
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
    print(f'trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # data_loader
    train_set = CrossDset(root_path, config['domain'], 'train', tokenizer, templates=task_subgroup_1, gid_dict=config['idx_name1'], sid_dict=config['idx_name2'], index_type=config['index_type'],
                          seed=config['seed'])
    val_set0 = CrossDset(root_path, config['domain'], 'val', tokenizer, templates=task_subgroup_1, gid_dict=config['idx_name1'], sid_dict=config['idx_name2'],
                         test_description_idx=0, index_type=config['index_type'])
    val_set1 = CrossDset(root_path, config['domain'], 'val', tokenizer, templates=task_subgroup_1, gid_dict=config['idx_name1'], sid_dict=config['idx_name2'],
                         test_description_idx=1, index_type=config['index_type'])
    val_set2 = CrossDset(root_path, config['domain'], 'val', tokenizer, templates=task_subgroup_1, gid_dict=config['idx_name1'], sid_dict=config['idx_name2'],
                         test_description_idx=2, index_type=config['index_type'])
    val_set3 = CrossDset(root_path, config['domain'], 'val', tokenizer, templates=task_subgroup_1, gid_dict=config['idx_name1'], sid_dict=config['idx_name2'],
                         test_description_idx=3, index_type=config['index_type'])
    # tset = CrossDset(root_path, domain, 'test', tokenizer, templates=task_subgroup_1, gid_dict=idx_name1, sid_dict=idx_name2, soft_prompt_len=0, description_idx=0)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=config['batch_size'],
                              collate_fn=train_set.collate_fn, num_workers=config['num_workers'])
    val_loader0 = DataLoader(val_set0, batch_size=config['test_batch_size'], collate_fn=val_set0.collate_fn, num_workers=config['num_workers'])
    val_loader1 = DataLoader(val_set1, batch_size=config['test_batch_size'], collate_fn=val_set1.collate_fn, num_workers=config['num_workers'])
    val_loader2 = DataLoader(val_set2, batch_size=config['test_batch_size'], collate_fn=val_set2.collate_fn, num_workers=config['num_workers'])
    val_loader3 = DataLoader(val_set3, batch_size=config['test_batch_size'], collate_fn=val_set3.collate_fn, num_workers=config['num_workers'])
    val_loaders = []
    if config['index_type'] == 'cross':
        val_loaders.append(val_loader0)
        val_loaders.append(val_loader1)
        val_loaders.append(val_loader2)
        val_loaders.append(val_loader3)
    elif config['index_type'] == 'both':
        val_loaders.append(val_loader0)
        val_loaders.append(val_loader1)
    elif config['index_type'] == 'gid':
        val_loaders.append(val_loader0)
    elif config['index_type'] == 'sid':
        val_loaders.append(val_loader0)
    else:
        raise NotImplementedError

    candidates_g = train_set.g_items
    candidate_trie_g = Trie([[0] + tokenizer.encode(candidate) for candidate in candidates_g])
    prefix_allowed_tokens_g = prefix_allowed_tokens_fn(candidate_trie_g)

    candidates_s = train_set.s_items
    candidate_trie_s = Trie([[0] + tokenizer.encode(candidate) for candidate in candidates_s])
    prefix_allowed_tokens_s = prefix_allowed_tokens_fn(candidate_trie_s)

    # train
    optim, lr_scheduler = create_optimizer_and_scheduler(config, train_loader, model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt_path = os.path.join(save_dir, f'model_best_0ep.pth')
    model.to(device)
    optim.zero_grad()
    best_ndcg = 0
    for epoch in range(config['epoch']):
        model.train()
        pbar = tqdm(train_loader, desc="Train epoch {}".format(epoch + 1))
        train_loss = 0.
        for step, batch in enumerate(pbar):
            results = model.train_step(batch)
            loss = results['loss']
            train_loss += loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_scheduler.step()
            optim.zero_grad()
            lr = lr_scheduler.get_last_lr()[0]
            pbar.set_postfix({'loss': loss.item(), 'lr': lr})
        print(f'{epoch + 1} train loss: {train_loss / step}, last_lr: {lr}')
        print('###############################################################')
        wandb.log({"train_loss": train_loss / step, 'epoch': epoch})
        wandb.log({"lr": lr, 'epoch': epoch})
        # validation
        print('evaluation ')
        model.eval()
        with torch.no_grad():
            mean_hit_5 = 0
            mean_hit_10 = 0
            mean_ndcg_5 = 0
            mean_ndcg_10 = 0
            for loader_idx, loader in enumerate(val_loaders):
                correct_validation_1 = 0
                correct_validation_5 = 0
                correct_validation_10 = 0
                ndcg_validation_5 = 0
                ndcg_validation_10 = 0
                validation_total = 0
                val_loss = 0
                if config['index_type'] == 'gid':
                    constraint = prefix_allowed_tokens_g
                elif config['index_type'] == 'sid':
                    constraint = prefix_allowed_tokens_s
                else:
                    if loader_idx % 2 == 0:
                        constraint = prefix_allowed_tokens_g
                    else:
                        constraint = prefix_allowed_tokens_s

                for step_i, batch in tqdm(enumerate(val_loaders[loader_idx])):
                    results = model.valid_step(batch)
                    loss = results['loss']
                    val_loss += loss
                    (one_hit_1, one_hit_5, one_hit_10, one_ndcg_5, one_ndcg_10, corr_list) = \
                        predict_outputs(batch, model, constraint, k=10, max_len=20, tokenizer=tokenizer)
                    correct_validation_1 += one_hit_1
                    correct_validation_5 += one_hit_5
                    correct_validation_10 += one_hit_10
                    ndcg_validation_5 += one_ndcg_5
                    ndcg_validation_10 += one_ndcg_10
                    validation_total += batch['input_ids'].size(0)
                hit_1 = round(correct_validation_1 / validation_total, 4)
                hit_5 = round(correct_validation_5 / validation_total, 4)
                hit_10 = round(correct_validation_10 / validation_total, 4)
                ndcg_5 = round(ndcg_validation_5 / validation_total, 4)
                ndcg_10 = round(ndcg_validation_10 / validation_total, 4)
                mean_hit_5 += hit_5
                mean_hit_10 += hit_10
                mean_ndcg_5 += ndcg_5
                mean_ndcg_10 += ndcg_10
                print(f'validation_{loader_idx}: {epoch + 1} val loss: {val_loss / step_i}')
                wandb.log({"val_loss": val_loss / step_i, 'epoch': epoch})
                print(f'hit@1: {hit_1}, hit@5: {hit_5}, hit@10: {hit_10}, ndcg@5: {ndcg_5}, ndcg@10: {ndcg_10}')
                print('###############################################################')
            hit_5 = mean_hit_5 / len(val_loaders)
            hit_10 = mean_hit_10 / len(val_loaders)
            ndcg_5 = mean_ndcg_5 / len(val_loaders)
            ndcg_10 = mean_ndcg_10 / len(val_loaders)
            wandb.log({"hit@5": hit_5, 'epoch': epoch})
            wandb.log({"hit@10": hit_10, 'epoch': epoch})
            wandb.log({"ndcg@5": ndcg_5, 'epoch': epoch})
            wandb.log({"ndcg@10": ndcg_10, 'epoch': epoch})

            if ndcg_10 > best_ndcg:
                best_ndcg = ndcg_10
                if os.path.isfile(ckpt_path):
                    os.remove(ckpt_path)
                torch.save(model, f'{save_dir}model_best_{epoch + 1}ep.pth')
                ckpt_path = os.path.join(save_dir, f'model_best_{epoch + 1}ep.pth')
    run.finish()
