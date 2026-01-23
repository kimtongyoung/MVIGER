import os
import pdb
from torch.utils.data.dataset import Dataset
from collections import defaultdict
import json
import torch
import random
# from data_loader.sampling_tree import *
from data_loader.prompt_p5 import task_subgroup_1 as prompt_templates


class MVIGERDset(Dataset):
    def __init__(self, data_path, domain, train_mode, tokenizer, tokenizer_prior, prompt_templates, ceid_dict, seid_dict, num_templates=10, num_indexes=2, train_templates=1, use_inst=False):
        super().__init__()
        self.sequential_data = self.ReadLineFromFile(os.path.join(data_path, domain, 'sequential_data.txt'))
        self.use_inst = use_inst
        self.ceid_dict = self.load_json(os.path.join(data_path, domain, ceid_dict))
        self.seid_dict = self.load_json(os.path.join(data_path, domain, seid_dict))
        self.iid_ceid, self.iid_seid, self.ceid_iid, self.seid_iid, self.idx_iid = self.index_to_input_dict(self.ceid_dict, self.seid_dict)
        self.train_mode = train_mode
        self.num_templates = num_templates
        self.train_templates = train_templates
        self.templates = prompt_templates[:num_templates]
        self.num_indexes = num_indexes
        self.tokenizer = tokenizer
        self.tokenizer_prior = tokenizer_prior
        self.user_items = defaultdict()
        self.user_list = []
        self.train_seq = []
        self.test_idx = []
        self.whole_seq = []
        self.task_instruction = ['Given <CEID> Predict <CEID>', 'Given <SEID> Predict <SEID>']


        for line in self.sequential_data:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            self.user_list.append(user)
            self.train_seq.append(items[:-2])
            self.test_idx.append(items[-1])
            self.whole_seq.append(items)
            self.user_items[user] = items
        self.c_items = list(self.ceid_iid.keys())
        self.s_items = list(self.seid_iid.keys())
        self.all_items = list(self.idx_iid.keys())
        self.total_num = self.number_of_interactions()


    def index_to_input_dict(self, ceid_dict, seid_dict):
        iid_ceid = defaultdict()
        iid_seid = defaultdict()
        ceid_iid = defaultdict()
        seid_iid = defaultdict()
        idx_iid = defaultdict()
        for key, codes in ceid_dict.items():
            if self.use_inst:
                text = 'item_<CEID>'
            else:
                text = 'item_'
            for level, code in enumerate(codes):
                if type(code) == str:
                    code = code.replace('Leaf', '')
                text += f'<CEID_{level}_{code}>'
 
            iid_ceid[key] = text
            ceid_iid[text] = key
            idx_iid[text] = key
        for key, codes in seid_dict.items():
            if self.use_inst:
                text = 'item_<SEID>'
            else:
                text = 'item_'
            for level, code in enumerate(codes):
                if type(code) == str:
                    code = code.replace('Leaf', '')
                text += f'<SEID_{level}_{code}>'
     
            iid_seid[key] = text
            seid_iid[text] = key
            idx_iid[text] = key
        return iid_ceid, iid_seid, ceid_iid, seid_iid, idx_iid

    def number_of_interactions(self):
        self.user_code = []
        self.target_idx = []
        total_num = 0

        start_idx = 0
        for k, v in zip(self.user_list, self.train_seq):
            number = len(v[1:])   
            total_num += number
            for idx in range(start_idx, number):
                self.user_code.append(k)
                self.target_idx.append(idx + 1)
                # total_num += 1

        return total_num

    def __len__(self):
        if self.train_mode == 'train':
            return self.total_num
            # return len(self.whole_seq)
        elif self.train_mode == 'val':
            return len(self.whole_seq)
        elif self.train_mode == 'test':
            return len(self.whole_seq)
        elif self.train_mode == 'pretrain':
            # return self.total_num * len(self.templates)
            return self.total_num * len(self.task_instruction)
        else:
            raise NotImplementedError

    def load_json(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def ReadLineFromFile(self, path):
        lines = []
        with open(path, 'r') as fd:
            for line in fd:
                lines.append(line.rstrip('\n'))
        return lines

    def __getitem__(self, idx):
        out_dict = {}
        out_dict['h'] = {}
        out_dict['z'] = {}

        if self.train_mode == 'pretrain':
            index_idx=idx//self.total_num
            template = random.choice(self.templates)
            instruction = self.task_instruction[index_idx]
            position_index = idx % self.total_num
            user_idx = self.user_code[position_index]
            seq_idx = self.user_list.index(user_idx)
            target_idx = self.target_idx[position_index]
            whole_sequence = self.train_seq[seq_idx]
            target_item = whole_sequence[target_idx]
            purchase_history = whole_sequence[:target_idx]
            purchase_history = purchase_history[-20:]

            pred_type = instruction.split(' ')[-1]
            if pred_type == '<CEID>':
                convert_iid = self.iid_ceid
            else:
                convert_iid = self.iid_seid
            if template['input_first'] == 'user':
                 input_sent = template["source"].format(
                user_idx,
                ", ".join([convert_iid[item_idx] for item_idx in purchase_history]),
            )
            else:
                input_sent = template["source"].format(
                    ", ".join([convert_iid[item_idx] for item_idx in purchase_history]),
                    user_idx,
                )
            if self.use_inst:
                input_sent += instruction
            
            output_sent = convert_iid[target_item]
            inp_seq = self.tokenizer(input_sent)
            oup_seq = self.tokenizer(output_sent)
            input_id = inp_seq['input_ids']
            attn_mask = inp_seq['attention_mask']
            target_id = oup_seq['input_ids']
            tokenized_text = self.tokenizer.convert_ids_to_tokens(input_id)
            whole_item_ids = self.calculate_whole_word_ids(tokenized_text, input_id)
            
            out_dict['input_sent'] = input_sent
            out_dict['input_ids'] = torch.LongTensor(input_id)
            out_dict['attention_mask'] = torch.LongTensor(attn_mask)
            out_dict['output_sent'] = output_sent
            out_dict['target_ids'] = torch.LongTensor(target_id)
            out_dict['whole_item_ids'] = torch.LongTensor(whole_item_ids)
            out_dict['input_length'] = len(input_id)
            out_dict['target_length'] = len(target_id)
            return out_dict

        else:
            if self.train_mode == "train":
                position_index = idx % self.total_num
                user_idx = self.user_code[position_index]
                seq_idx = self.user_list.index(user_idx)
                target_idx = self.target_idx[position_index]
                whole_sequence = self.train_seq[seq_idx]
                target_item = whole_sequence[target_idx]
                purchase_history = whole_sequence[:target_idx]
                # sequence = self.whole_seq[idx]
                # user_idx = self.user_list[idx]
                # purchase_history = sequence[:-3]
                # target_item = sequence[-3]
            elif self.train_mode == 'val':
                sequence = self.whole_seq[idx]
                user_idx = self.user_list[idx]
                purchase_history = sequence[:-2]
                target_item = sequence[-2]
            elif self.train_mode == 'test':
                sequence = self.whole_seq[idx]
                user_idx = self.user_list[idx]
                purchase_history = sequence[:-1]
                target_item = sequence[-1]
            

            purchase_history = purchase_history[-20:]
            if self.use_inst:
                base_template = "<Prior> user_{} has purchased : item_<IID>{}, predict which view will be best?"
                output_sent = f'item_<IID><IID_{target_item}>'
            else:
                base_template = "<Prior> user_{} has purchased : item_{}, predict which view will be best?"
                output_sent = f'item_<IID_{target_item}>'
            
            input_sent = base_template.format(user_idx, ", ".join([f'<IID_{item_idx}>' for item_idx in purchase_history]))
            

            inp_seq = self.tokenizer_prior(input_sent)
            oup_seq = self.tokenizer_prior(output_sent)

            input_id = inp_seq['input_ids']
            attn_mask = inp_seq['attention_mask']
            target_id = oup_seq['input_ids']

            tokenized_text = self.tokenizer_prior.convert_ids_to_tokens(input_id)
            whole_item_ids = self.calculate_whole_word_ids(tokenized_text, input_id)

            out_dict['h']['input_sent'] = input_sent
            out_dict['h']['input_ids'] = torch.LongTensor(input_id)
            out_dict['h']['attention_mask'] = torch.LongTensor(attn_mask)
            out_dict['h']['output_sent'] = output_sent
            out_dict['h']['target_ids'] = torch.LongTensor(target_id)
            out_dict['h']['whole_item_ids'] = torch.LongTensor(whole_item_ids)
            out_dict['h']['input_length'] = len(input_id)
            out_dict['h']['target_length'] = len(target_id)

            # for z
            iid_view = [self.iid_ceid, self.iid_seid]
            out_dict['z']['input_sent'] = []
            out_dict['z']['input_ids'] = []
            out_dict['z']['attention_mask'] = []
            out_dict['z']['output_sent'] = []
            out_dict['z']['target_ids'] = []
            out_dict['z']['whole_item_ids'] = []
            out_dict['z']['input_length'] = []
            out_dict['z']['target_length'] = []

            for i in range(self.num_indexes):
                for t in range(self.num_templates):
                    if self.templates[t]["input_first"] == "user":
                        input_sent = self.templates[t]["source"].format(
                            user_idx,
                            ", ".join([iid_view[i][item_idx] for item_idx in purchase_history]),
                        )
                    else:
                        input_sent = self.templates[t]["source"].format(
                            ", ".join([iid_view[i][item_idx] for item_idx in purchase_history]),
                            user_idx,
                        )
                    output_sent = iid_view[i][target_item]

                    instruction_C= ' Given <CEID> Predict <CEID>.'
                    instruction_S= ' Given <SEID> Predict <SEID>.'

                    if self.use_inst:
                        if i == 0:
                            input_sent += instruction_C
                        else:
                            input_sent += instruction_S

                    inp_seq = self.tokenizer(input_sent)
                    oup_seq = self.tokenizer(output_sent)

                    input_id = inp_seq['input_ids']
                    attn_mask = inp_seq['attention_mask']
                    target_id = oup_seq['input_ids']

                    tokenized_text = self.tokenizer.convert_ids_to_tokens(input_id)
                    whole_item_ids = self.calculate_whole_word_ids(tokenized_text, input_id)
                    
                    out_dict['z']['input_sent'].append(input_sent)
                    out_dict['z']['input_ids'].append(torch.LongTensor(input_id))
                    out_dict['z']['attention_mask'].append(torch.LongTensor(attn_mask))
                    out_dict['z']['output_sent'].append(output_sent)
                    out_dict['z']['target_ids'].append(torch.LongTensor(target_id))
                    out_dict['z']['whole_item_ids'].append(torch.LongTensor(whole_item_ids))
                    out_dict['z']['input_length'].append(len(input_id))
                    out_dict['z']['target_length'].append(len(target_id))

            return out_dict

    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for token_idx in range(len(tokenized_text)):
            if tokenized_text[token_idx] == "<pad>":
                curr = 0
                whole_word_ids.append(curr)
            elif tokenized_text[token_idx].startswith("▁"):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        return whole_word_ids[: len(input_ids) - 1] + [0]

    def collate_fn(self, batch):
        batch_entry = {}
        if self.train_mode == 'pretrain':
            max_input_length = max(entry['input_length'] for entry in batch)
            max_target_length = max(entry['target_length'] for entry in batch)
            B = len(batch)

            input_ids = torch.zeros(B, max_input_length, dtype=torch.long)
            attention_mask = torch.zeros(B, max_input_length, dtype=torch.long)
            whole_item_ids = torch.zeros(B, max_input_length, dtype=torch.long)

            target_ids = torch.ones(B, max_target_length, dtype=torch.long) * -100
            target_length = torch.zeros(B, 1, dtype=torch.long)

            for batch_idx, entry in enumerate(batch):
                input_ids[batch_idx, :entry['input_length']] = entry['input_ids']
                attention_mask[batch_idx, :entry['input_length']] = entry['attention_mask']
                whole_item_ids[batch_idx, :entry['input_length']] = entry['whole_item_ids']

                target_ids[batch_idx, :entry['target_length']] = entry['target_ids']
                target_length[batch_idx] = entry['target_length']

            batch_entry['input_ids'] = input_ids
            batch_entry['attention_mask'] = attention_mask
            batch_entry['whole_item_ids'] = whole_item_ids
            batch_entry['target_ids'] = target_ids
            batch_entry['target_length'] = target_length

        else:
            batch_entry['h'] = {}
            batch_entry['z'] = {}

            B = len(batch)
            max_input_length = max(entry['h']['input_length'] for entry in batch)
            max_target_length = max(entry['h']['target_length'] for entry in batch)

            h_input_ids = torch.zeros(B, max_input_length, dtype=torch.long)
            h_attention_mask = torch.zeros(B, max_input_length, dtype=torch.long)
            h_whole_item_ids = torch.zeros(B, max_input_length, dtype=torch.long)
            h_target_ids = torch.ones(B, max_target_length, dtype=torch.long) * -100
            h_target_length = torch.zeros(B, 1, dtype=torch.long)

            for batch_idx, entry in enumerate(batch):
                h_input_ids[batch_idx, :entry['h']['input_length']] = entry['h']['input_ids']
                h_attention_mask[batch_idx, :entry['h']['input_length']] = entry['h']['attention_mask']
                h_whole_item_ids[batch_idx, :entry['h']['input_length']] = entry['h']['whole_item_ids']
                h_target_ids[batch_idx, :entry['h']['target_length']] = entry['h']['target_ids']
                h_target_length[batch_idx] = entry['h']['target_length']

            batch_entry['h']['input_ids'] = h_input_ids
            batch_entry['h']['attention_mask'] = h_attention_mask
            batch_entry['h']['whole_item_ids'] = h_whole_item_ids
            batch_entry['h']['target_ids'] = h_target_ids
            batch_entry['h']['target_length'] = h_target_length

            # for z
            max_input_length = max(max(entry['z']['input_length']) for entry in batch)
            max_target_length = max(max(entry['z']['target_length']) for entry in batch)

            z_input_ids = torch.zeros(B * self.num_indexes * self.num_templates, max_input_length, dtype=torch.long)
            z_attention_mask = torch.zeros(B * self.num_indexes * self.num_templates, max_input_length, dtype=torch.long)
            z_whole_item_ids = torch.zeros(B * self.num_indexes * self.num_templates, max_input_length, dtype=torch.long)
            z_target_ids = torch.ones(B * self.num_indexes * self.num_templates, max_target_length, dtype=torch.long) * -100
            z_target_length = torch.zeros(B * self.num_indexes * self.num_templates, 1, dtype=torch.long)

            
            for batch_idx, entry in enumerate(batch):
                for view_idx in range(self.num_indexes*self.num_templates):
                    z_input_ids[batch_idx*self.num_indexes*self.num_templates+view_idx, :entry['z']['input_length'][view_idx]] = entry['z']['input_ids'][view_idx]
                    z_attention_mask[batch_idx*self.num_indexes*self.num_templates+view_idx, :entry['z']['input_length'][view_idx]] = entry['z']['attention_mask'][view_idx]
                    z_whole_item_ids[batch_idx*self.num_indexes*self.num_templates+view_idx, :entry['z']['input_length'][view_idx]] = entry['z']['whole_item_ids'][view_idx]
                    z_target_ids[batch_idx*self.num_indexes*self.num_templates+view_idx, :entry['z']['target_length'][view_idx]] = entry['z']['target_ids'][view_idx]
                    z_target_length[batch_idx*self.num_indexes*self.num_templates+view_idx] = entry['z']['target_length'][view_idx]

            batch_entry['z']['input_ids'] = z_input_ids
            batch_entry['z']['attention_mask'] = z_attention_mask
            batch_entry['z']['whole_item_ids'] = z_whole_item_ids
            batch_entry['z']['target_ids'] = z_target_ids
            batch_entry['z']['target_length'] = z_target_length

        return batch_entry

