import os
import pdb
from torch.utils.data.dataset import Dataset
from collections import defaultdict
import json
import torch
import random


class SCRecDset(Dataset):
    def __init__(self, data_path, data_name, train_mode='train', tokenizer=None, templates=None, ceid_dict='', seid_dict='', num_templates=10, test_instruction_type='ceid', seed=2024):
        super().__init__()
        random.seed(seed)
        self.sequential_data = self.ReadLineFromFile(os.path.join(data_path, data_name, 'sequential_data.txt'))
        self.ceid_dict = self.load_json(os.path.join(data_path, data_name, ceid_dict))
        self.seid_dict = self.load_json(os.path.join(data_path, data_name, seid_dict))
        self.iid_ceid, self.iid_seid, self.ceid_iid, self.seid_iid, self.idx_iid = self.create_index_dict(self.ceid_dict, self.seid_dict)
        self.train_mode = train_mode
        self.num_templates = num_templates
        self.templates = templates[:num_templates]
        self.tokenizer = tokenizer
        self.item_count = defaultdict(int)
        self.user_items = defaultdict()
        self.user_list = []
        self.train_seq = []
        self.whole_seq = []
        self.task_instruction = ['Given <G> Predict <G>', 'Given <S> Predict <S>']
        if test_instruction_type == 'ceid':
            self.test_instruction_idx = 0
        elif test_instruction_type == 'seid':
            self.test_instruction_idx = 1
        else:
            raise NotImplementedError
        for line in self.sequential_data:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            self.user_list.append(user)
            self.train_seq.append(items[:-2])
            self.whole_seq.append(items)
            self.user_items[user] = items
        self.c_items = list(self.ceid_iid.keys())
        self.s_items = list(self.seid_iid.keys())
        self.all_items = list(self.idx_iid.keys())
        self.total_num = self.number_of_interactions()

    def create_index_dict(self, ceid_dict, seid_dict):
        iid_ceid = defaultdict()
        iid_seid = defaultdict()
        ceid_iid = defaultdict()
        seid_iid = defaultdict()
        idx_iid = defaultdict()
        for key, codes in ceid_dict.items():
            text = 'item_<C>'
            for level, code in enumerate(codes):
                text += f'<extra_c_{level}_{code}>'
            iid_ceid[key] = text
            ceid_iid[text] = key
            idx_iid[text] = key
        for key, codes in seid_dict.items():
            text = 'item_<S>'
            for level, code in enumerate(codes):
                text += f'<extra_s_{level}_{code}>'
            iid_seid[key] = text
            seid_iid[text] = key
            idx_iid[text] = key
        return iid_ceid, iid_seid, ceid_iid, seid_iid, idx_iid

    def number_of_interactions(self):
        self.user_code = []
        self.target_idx = []
        total_num = 0
        for k, v in zip(self.user_list, self.train_seq):
            number = len(v[1:])
            total_num += number
            for idx in range(number):
                self.user_code.append(k)
                self.target_idx.append(idx + 1)
        return total_num

    def __len__(self):
        if self.train_mode == 'train':
            return self.total_num * len(self.templates)
        elif self.train_mode == 'val':
            return len(self.whole_seq)
        elif self.train_mode == 'test':
            return len(self.whole_seq)
        elif self.train_mode == 'infer':
            return len(self.whole_seq) * len(self.templates)
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
        if self.train_mode == "train":
            instruction = random.choice(self.task_instruction)
            template_idx = idx // self.total_num
            template = self.templates[template_idx]
            position_index = idx % self.total_num
            user_idx = self.user_code[position_index]
            seq_idx = self.user_list.index(user_idx)
            target_idx = self.target_idx[position_index]
            whole_sequence = self.train_seq[seq_idx]
            target_item = whole_sequence[target_idx]
            purchase_history = whole_sequence[:target_idx]
        elif self.train_mode == 'val':
            instruction = self.task_instruction[self.test_instruction_idx]
            template = self.templates[0]
            sequence = self.whole_seq[idx]
            user_idx = self.user_list[idx]
            purchase_history = sequence[:-2]
            target_item = sequence[-2]
        elif self.train_mode == 'test':
            instruction = self.task_instruction[self.test_instruction_idx]
            template = self.templates[0]
            sequence = self.whole_seq[idx]
            user_idx = self.user_list[idx]
            purchase_history = sequence[:-1]
            target_item = sequence[-1]
        elif self.train_mode == 'infer':
            template_idx = idx % len(self.templates)
            template = self.templates[template_idx]
            instruction = self.task_instruction[self.test_instruction_idx]
            idx = idx // len(self.templates)
            sequence = self.whole_seq[idx]
            user_idx = self.user_list[idx]
            purchase_history = sequence[:-1]
            target_item = sequence[-1]
        else:
            raise NotImplementedError

        purchase_history = purchase_history[-20:]

        _, given_type, _, pred_type = instruction.split(' ')
        if given_type == '<G>':
            convert_given_iid = self.iid_ceid
        else:
            convert_given_iid = self.iid_seid
        if pred_type == '<G>':
            convert_pred_iid = self.iid_ceid
        else:
            convert_pred_iid = self.iid_seid
        if template["input_first"] == "user":
            input_sent = template["source"].format(
                user_idx,
                ", ".join([convert_given_iid[item_idx] for item_idx in purchase_history]),
            )
        else:
            input_sent = template["source"].format(
                ", ".join([convert_given_iid[item_idx] for item_idx in purchase_history]),
                user_idx,
            )

        input_sent += instruction
        output_sent = convert_pred_iid[target_item]

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
        B = len(batch)
        max_input_length = max(entry['input_length'] for entry in batch)
        max_target_length = max(entry['target_length'] for entry in batch)

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
        return batch_entry

