import argparse
import collections
from parse_config import ConfigParser
from torch.utils.data.dataset import Dataset
from sentence_transformers import SentenceTransformer
import os
import json
import re
from collections import defaultdict
import tqdm
try:
    from pickle5 import pickle
except:
    from pickle import pickle


class SentenceEmbeddingDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.datamap = self.load_json(os.path.join('data', config['domain'], 'datamaps.json'))[
            'id2item']
        self.meta_dict = self.load_json(os.path.join('data', config['domain'], 'meta_data.json'))
        self.item_keys = list(self.datamap.keys())
        self.model = SentenceTransformer('t5-base').cpu()
        if config['domain'] == 'yelp':
            self.is_yelp = True
        else:
            self.is_yelp = False
    def __len__(self):
        return len(self.datamap)

    def load_json(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    # remove html tags and escape sid
    def text_cleaning(self, text):
        text = re.sub(r'</?\w+[^>]*>', '', text)
        text = re.sub(r'["\n\r]*', '', text)
        return text

    def __getitem__(self, idx):
        asin = self.datamap[str(idx)]
        meta = self.meta_dict[asin]
        text = ''
        if self.is_yelp:
            if 'name' in meta.keys():
                text += f'{meta["name"]}, '
            if 'city' in meta.keys():
                text += f'{meta["city"]}, '
            if 'state' in meta.keys():
                text += f'{meta["state"]}, '
            if 'attributes' in meta.keys():
                if meta['attributes'] is not None:
                    for att, value in meta['attributes'].items():
                        value = eval(value)
                        if value == False:
                            continue
                        elif value == True:
                            text += f'{att}, '
                        elif type(value) == dict:
                            for k, v in value.items():
                                if v == True:
                                    text += f'{k}, '
                        else:
                            text += f'{att}: {value}, '

        else:
            if 'title' in meta.keys():
                text += f'{meta["title"]} '
            if 'brand' in meta.keys():
                text += f'{meta["brand"]} '
            if 'categories' in meta.keys():
                text += f'{meta["categories"][0]} |'
            if 'description' in meta.keys():
                text += f'{meta["description"]}'
        text = self.text_cleaning(text)
        out_dict = defaultdict()
        out_dict['text'] = text
        embedding = self.model.encode(text)
        out_dict['embedding'] = embedding
        out_dict['idx'] = idx
        return out_dict


def get_embedding(dset):
    result = defaultdict()
    for idx in tqdm.tqdm(range(len(dset))):
        result[dset[idx]['idx']] = dset[idx]['embedding']
    return result


def save_embedding(root_path, domain, result_dict, name):
    with open(os.path.join(root_path, domain, f'{name}.pkl'), 'wb') as fOut:
        pickle.dump(result_dict, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    return print('saving embedding done!')


def load_embedding(root_path, domain, name):
    with open(os.path.join(root_path, domain, f'{name}.pkl'), 'rb') as fIn:
        data = pickle.load(fIn)
    return data


def create_embedding(config):
    dset = SentenceEmbeddingDataset(config)
    res = get_embedding(dset)
    save_embedding('data', config['domain'], res, 't5-base')
    print(f"Semantic Embedding is saved at : {os.path.join('data', config['domain'], 't5-base')}")



##########################################

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='./config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    create_embedding(config)
