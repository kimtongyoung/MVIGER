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
from data_loader.mviger_loader import MVIGERDset
from data_loader.prompt_p5 import task_subgroup_1
import os
from transformers import T5Config
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.optim import AdamW
import json
from collections import Counter

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
    # warmup_iters = steps_per_epoch * 10
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


def mviger_runner(config):
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

        model.resize_token_embeddings(t5_config.vocab_size)
    print(f'trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if config['pretrained_path'] != "":
        state_dict = torch.load(config['pretrained_path'], 'cpu').state_dict()
        if config['stage'] == 1:
            for k in ["h_embed.weight", "encoder_prior.embed_tokens.weight"]:
                if k in state_dict:
                    del state_dict[k]
        if config['stage'] == 0:
            for k in ["shared.weight", "lm_head.weight", "decoder.embed_tokens.weight", "encoder.embed_tokens.weight", "h_embed.weight", "encoder_prior.embed_tokens.weight"]:
                if k in state_dict:
                    del state_dict[k]
        model.load_state_dict(state_dict, strict=False)

    prompt = task_subgroup_1

    # data_loader
    if config['stage'] == 0:
        train_set = MVIGERDset(root_path, config['domain'], 'pretrain', tokenizer, tokenizer_prior, prompt_templates=prompt, ceid_dict=config['idx_name1'], seid_dict=config['idx_name2'],
                              num_templates=config['num_templates'], num_indexes=config['num_indexes'], train_templates=config['train_templates'], use_inst=config['use_inst'])
    else:
        train_set = MVIGERDset(root_path, config['domain'], 'train', tokenizer, tokenizer_prior, prompt_templates=prompt, ceid_dict=config['idx_name1'], seid_dict=config['idx_name2'],
                          num_templates=config['num_templates'], num_indexes=config['num_indexes'], use_inst=config['use_inst'])
    val_set = MVIGERDset(root_path, config['domain'], 'val', tokenizer, tokenizer_prior, prompt_templates=prompt, ceid_dict=config['idx_name1'], seid_dict=config['idx_name2'],
                         num_templates=config['num_templates'], num_indexes=config['num_indexes'], use_inst=config['use_inst'])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=config['batch_size'], collate_fn=train_set.collate_fn, num_workers=config['num_workers'])
    
    val_loader = DataLoader(val_set, batch_size=config['test_batch_size'], collate_fn=val_set.collate_fn, num_workers=config['num_workers'])


    candidates_c = train_set.c_items
    candidate_trie_c = Trie([[0] + tokenizer.encode(candidate) for candidate in candidates_c])
    prefix_allowed_tokens_c = prefix_allowed_tokens_fn(candidate_trie_c)

    candidates_s = train_set.s_items
    candidate_trie_s = Trie([[0] + tokenizer.encode(candidate) for candidate in candidates_s])
    prefix_allowed_tokens_s = prefix_allowed_tokens_fn(candidate_trie_s)

    constraints = [prefix_allowed_tokens_c, prefix_allowed_tokens_s]

    # train
    optim, lr_scheduler = create_optimizer_and_scheduler(config, train_loader, model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    optim.zero_grad()
    best_hit = 0
    min_val_loss = 100
    for epoch in range(config['epoch']):
        model.train()
        
        pbar = tqdm(train_loader, desc="Train epoch {}".format(epoch + 1))

        total_loss = 0.

        for step, batch in enumerate(pbar):
            results = model.train_step(batch, stage=config['stage'], k=config['beam_size'], tokenizer=tokenizer, constraints=constraints, 
            use_ll_norm=config['use_ll_norm'], pooling_type=config['pooling_type'], alpha = config['alpha'], beta = config['beta'], tau=config['tau'])
            lr = lr_scheduler.get_last_lr()[0]
            loss = results['total_loss']
            if config['stage'] != 0:
                pbar.set_postfix({'loss': loss.item(), 'kl': results['kl'].item(), 'h_p': results['h_p'].item(), 'h_q': results['h_q'].item(), 'lr': lr})
            else:
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
       

        
        val_start = config['val_start']
        if epoch >=val_start:
            if epoch % config['val_interval'] == 0: 
                model.eval()
                with torch.no_grad():
                    if config['stage'] == 0:
                        hit_5_pred = 0
                        hit_10_pred = 0
                        ndcg_5_pred = 0
                        ndcg_10_pred = 0
                        ceid_hit_5 = 0
                        seid_hit_5 = 0
                        ceid_hit_10 = 0
                        seid_hit_10 = 0
                        ceid_ndcg_5 = 0
                        seid_ndcg_5 = 0
                        ceid_ndcg_10 = 0
                        seid_ndcg_10 = 0
                        validation_total = 0
                        val_loss_total = 0.
                    
                        pbar = tqdm(val_loader, desc="Val epoch {}".format(epoch + 1))
                        for step, batch in enumerate(pbar):
                            results = model.valid_pretrain(batch, k=10, max_len=20, constraints=constraints, tokenizer=tokenizer)
                            hit_1, hit_5, hit_10, ndcg_5, ndcg_10, i_pred, t_pred, hit_1_ceid, hit_5_ceid, hit_10_ceid, ndcg_5_ceid, ndcg_10_ceid, hit_1_seid, hit_5_seid, hit_10_seid, ndcg_5_seid, ndcg_10_seid = results
                            hit_5_pred += hit_5
                            hit_10_pred += hit_10
                            ndcg_5_pred += ndcg_5
                            ndcg_10_pred += ndcg_10
                            ceid_hit_5 += hit_5_ceid
                            seid_hit_5 += hit_5_seid
                            ceid_hit_10 += hit_10_ceid
                            seid_hit_10 += hit_10_seid
                            ceid_ndcg_5 += ndcg_5_ceid
                            seid_ndcg_5 += ndcg_5_seid
                            ceid_ndcg_10 += ndcg_10_ceid
                            seid_ndcg_10 += ndcg_10_seid
                                
                            B = batch['h']['input_ids'].size(0)
                            validation_total += B
                        hit_5_pred = round(hit_5_pred / validation_total, 4)
                        hit_10_pred = round(hit_10_pred / validation_total, 4)
                        ndcg_5_pred = round(ndcg_5_pred / validation_total, 4)
                        ndcg_10_pred = round(ndcg_10_pred / validation_total, 4)
                        ceid_hit_5 = round(ceid_hit_5 / validation_total, 4)
                        seid_hit_5 = round(seid_hit_5 / validation_total, 4)
                        ceid_hit_10 = round(ceid_hit_10 / validation_total, 4)
                        seid_hit_10 = round(seid_hit_10 / validation_total, 4)
                        ceid_ndcg_5 = round(ceid_ndcg_5 / validation_total, 4)
                        seid_ndcg_5 = round(seid_ndcg_5 / validation_total, 4)
                        ceid_ndcg_10 = round(ceid_ndcg_10 / validation_total, 4)
                        seid_ndcg_10 = round(seid_ndcg_10 / validation_total, 4)
                        print(f'hit@5_ceid: {ceid_hit_5}, hit@5_seid: {seid_hit_5}, hit@10_ceid: {ceid_hit_10}, hit@10_seid: {seid_hit_10}')
                        print(f'ndcg@5_ceid: {ceid_ndcg_5}, ndcg@5_seid: {seid_ndcg_5}, ndcg@10_ceid: {ceid_ndcg_10}, ndcg@10_seid: {seid_ndcg_10}')
                        
                        print('###############################################################')

                        wandb.log({f"val_hit@5_ceid": ceid_hit_5, 'epoch': epoch})
                        wandb.log({f'val_hit@5_seid': seid_hit_5, 'epoch': epoch})
                        wandb.log({f'val_hit@10_ceid': ceid_hit_10, 'epoch': epoch})
                        wandb.log({f'val_hit@10_seid': seid_hit_10, 'epoch': epoch})
                        wandb.log({f"val_ndcg@5_ceid": ceid_ndcg_5, 'epoch': epoch})
                        wandb.log({f"val_ndcg@5_seid": seid_ndcg_5, 'epoch': epoch})
                        wandb.log({f"val_ndcg@10_ceid": ceid_ndcg_10, 'epoch': epoch})
                        wandb.log({f"val_ndcg@10_seid": seid_ndcg_10, 'epoch': epoch})

                    
                        avg_hit = (ceid_hit_10+seid_hit_10)/2
                        if avg_hit >= best_hit:
                            best_hit = avg_hit
                            torch.save(model, f'{save_dir}model_best_{epoch + 1}ep.pth')
                    else:  
                        total_loss = 0.
                        for val_step, batch in enumerate(pbar):

                            results = model.train_step(batch, stage=config['stage'], k=config['beam_size'], tokenizer=tokenizer, constraints=constraints, 
                                        use_ll_norm=config['use_ll_norm'], pooling_type=config['pooling_type'], alpha = config['alpha'], beta = config['beta'], tau=config['tau'])
                            loss = results['total_loss']
                            total_loss += loss
                            pbar.set_postfix({'loss': loss.item(), 'kl': results['kl'].item(), 'h_p': results['h_p'].item(), 'h_q': results['h_q'].item()})
                        avg_val_loss = total_loss / val_step

                        wandb.log({f"val_loss": avg_val_loss, 'epoch': epoch})

                        if avg_val_loss < min_val_loss:
                            min_val_loss = avg_val_loss
                            torch.save(model, f'{save_dir}model_best_{epoch + 1}ep.pth')
                            

    run.finish()