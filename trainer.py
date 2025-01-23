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
from data_loader.amazon_loader import PCRecDset
from prompt_p5 import task_subgroup_1 as p5_prompt
from prompt_ours import task_subgroup_1 as our_prompt
import os
from transformers import T5Config
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "100"


def create_model(model_class, backbone, t5_config=None):
    model = model_class.from_pretrained(backbone, config=t5_config)
    return model


def create_optimizer_and_scheduler(config, train_loader, model):
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
    optim = AdamW(optimizer_grouped_parameters, lr=config['lr'])
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
    save_dir = f'./saved/models/{config["project_name"]}/{save_name}/'

    print(f'training: {save_name}')
    ############## wandb logging
    run = wandb.init(project=config['project_name'], reinit=True)
    now = datetime.now()
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
    print(f'trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if config['prompt'] == 'p5':
        prompt = p5_prompt
    else:
        prompt = our_prompt

    # data_loader
    train_set = PCRecDset(root_path, config['domain'], 'train', tokenizer, templates=prompt, ceid_dict=config['idx_name1'], seid_dict=config['idx_name2'],
                          seed=config['seed'], is_p5id=config['is_p5id'])
    val_set0 = PCRecDset(root_path, config['domain'], 'val', tokenizer, templates=prompt, ceid_dict=config['idx_name1'], seid_dict=config['idx_name2'],
                         test_instruction_type='ceid', is_p5id=config['is_p5id'])
    val_set1 = PCRecDset(root_path, config['domain'], 'val', tokenizer, templates=prompt, ceid_dict=config['idx_name1'], seid_dict=config['idx_name2'],
                         test_instruction_type='seid', is_p5id=config['is_p5id'])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=config['batch_size'], collate_fn=train_set.collate_fn, num_workers=config['num_workers'])
    val_loader0 = DataLoader(val_set0, batch_size=config['test_batch_size'], collate_fn=val_set0.collate_fn, num_workers=config['num_workers'])
    val_loader1 = DataLoader(val_set1, batch_size=config['test_batch_size'], collate_fn=val_set1.collate_fn, num_workers=config['num_workers'])
    val_loaders = []
    val_loaders.append(val_loader0)
    val_loaders.append(val_loader1)

    candidates_c = train_set.c_items
    candidate_trie_c = Trie([[0] + tokenizer.encode(candidate) for candidate in candidates_c])
    prefix_allowed_tokens_c = prefix_allowed_tokens_fn(candidate_trie_c)

    candidates_s = train_set.s_items
    candidate_trie_s = Trie([[0] + tokenizer.encode(candidate) for candidate in candidates_s])
    prefix_allowed_tokens_s = prefix_allowed_tokens_fn(candidate_trie_s)

    # train
    optim, lr_scheduler = create_optimizer_and_scheduler(config, train_loader, model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt_path = os.path.join(save_dir, f'model_best_0ep.pth')
    model.to(device)
    optim.zero_grad()
    best_hit = 0
    for epoch in range(config['epoch']):
        model.train()
        pbar = tqdm(train_loader, desc="Train epoch {}".format(epoch + 1))
        total_loss = 0.
        base_loss = 0.
        bpr_loss = 0.
        for step, batch in enumerate(pbar):
            results = model.train_step(batch, config['bpr'])
            lr = lr_scheduler.get_last_lr()[0]
            if config['bpr']:
                loss = results['loss'] + results['bpr_loss'] * config['bpr_temp']
                pbar.set_postfix({'loss': loss.item(), 'bpr': results['bpr_loss'].item(), 'lr': lr})
                base_loss += results['loss']
                bpr_loss += results['bpr_loss']
            else:
                loss = results['loss']
                pbar.set_postfix({'loss': loss.item(), 'lr': lr})
            total_loss += loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_scheduler.step()
            optim.zero_grad()

        print(f'{epoch + 1} total loss: {total_loss / step}, last_lr: {lr}')
        print('###############################################################')
        wandb.log({"total_loss": total_loss / step, 'epoch': epoch})
        wandb.log({"lr": lr, 'epoch': epoch})
        if config['bpr']:
            wandb.log({"base_loss": base_loss / step, 'epoch': epoch})
            wandb.log({"bpr_loss": bpr_loss / step, 'epoch': epoch})

        # validation
        model.eval()
        with torch.no_grad():
            avg_hit_10 = 0
            for loader_idx, loader in enumerate(val_loaders):
                correct_validation_5 = 0
                correct_validation_10 = 0
                ndcg_validation_5 = 0
                ndcg_validation_10 = 0
                validation_total = 0
                val_loss = 0
                if loader_idx % 2 == 0:
                    constraint = prefix_allowed_tokens_c
                else:
                    constraint = prefix_allowed_tokens_s

                for step_i, batch in tqdm(enumerate(val_loaders[loader_idx])):
                    results = model.valid_step(batch)
                    loss = results['loss']
                    val_loss += loss
                    (one_hit_1, one_hit_5, one_hit_10, one_ndcg_5, one_ndcg_10, corr_list) = \
                        predict_outputs(batch, model, constraint, k=10, max_len=20, tokenizer=tokenizer)
                    correct_validation_5 += one_hit_5
                    correct_validation_10 += one_hit_10
                    ndcg_validation_5 += one_ndcg_5
                    ndcg_validation_10 += one_ndcg_10

                    validation_total += batch['input_ids'].size(0)

                hit_5 = round(correct_validation_5 / validation_total, 4)
                hit_10 = round(correct_validation_10 / validation_total, 4)
                ndcg_5 = round(ndcg_validation_5 / validation_total, 4)
                ndcg_10 = round(ndcg_validation_10 / validation_total, 4)

                avg_hit_10 += hit_10
                print(f'validation_{loader_idx}: {epoch + 1} val loss: {val_loss / step_i}')
                print(f'hit@5: {hit_5}, hit@10: {hit_10}, ndcg@5: {ndcg_5}, ndcg@10: {ndcg_10}')
                print('###############################################################')
                wandb.log({f"val_loss_{loader_idx}": val_loss / step_i, 'epoch': epoch})
                wandb.log({f"hit@5_{loader_idx}": hit_5, 'epoch': epoch})
                wandb.log({f"hit@10_{loader_idx}": hit_10, 'epoch': epoch})
                wandb.log({f"ndcg@5_{loader_idx}": ndcg_5, 'epoch': epoch})
                wandb.log({f"ndcg@10_{loader_idx}": ndcg_10, 'epoch': epoch})

            avg_hit_10 = avg_hit_10 / len(val_loaders)

            if avg_hit_10 > best_hit:
                best_hit = avg_hit_10
                # if os.path.isfile(ckpt_path):
                #     os.remove(ckpt_path)
                torch.save(model, f'{save_dir}model_best_{epoch + 1}ep.pth')
                # ckpt_path = os.path.join(save_dir, f'model_best_{epoch + 1}ep.pth')
    run.finish()
