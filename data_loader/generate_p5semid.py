import pdb
import random
import json
import os
from collections import defaultdict, Counter

# domain = 'Beauty'
# domain = 'Sports_and_Outdoors'
domain = 'Yelp'

path = f'data/amazon/filtered/{domain}/'


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


########### data preprocessing ###########
with open(path + "sequential_data.txt", "r") as f:
    data = f.read()

data = data.split("\n")[:-1]
data = [d.split(" ")[1:-2] for d in data]

data_map = load_json(os.path.join(path, 'datamaps.json'))
meta_data = load_json(os.path.join(path, 'meta_data.json'))

if domain == 'Yelp':
    leaf_index_count = {}
    final_data = {}
    vocab = []
    for iid in range(len(data_map['id2item'])):
        asin = data_map['id2item'][str(iid)]
        category_data = meta_data[asin]['categories']
        try:
            category_data = category_data.split(', ')
            for cate in category_data:
                vocab.append(cate)
        except:
            print(iid, category_data)
            category_data = ['None']
            vocab.append('None')

        if tuple(category_data) not in leaf_index_count:
            leaf_index_count[tuple(category_data)] = 0
        else:
            leaf_index_count[tuple(category_data)] += 1
        vocab.append("Leaf" + str(leaf_index_count[tuple(category_data)]))
        final_idx = category_data + ["Leaf" + str(leaf_index_count[tuple(category_data)])]
        final_data[iid] = final_idx
    vocab = list(set(vocab))

else:
    leaf_index_count = {}
    final_data = {}
    vocab = []

    for iid in range(len(data_map['id2item'])):
        asin = data_map['id2item'][str(iid)]
        category_data = meta_data[asin]['categories'][0]
        if category_data == ['']:
            vocab.append('None')
            print(iid)
        elif category_data == []:
            vocab.append('None')
            print(iid)
        else:
            for cate in category_data:
                vocab.append(cate)
        if tuple(category_data) not in leaf_index_count:
            leaf_index_count[tuple(category_data)] = 0
        else:
            leaf_index_count[tuple(category_data)] += 1
        vocab.append("Leaf" + str(leaf_index_count[tuple(category_data)]))
        final_idx = category_data + ["Leaf" + str(leaf_index_count[tuple(category_data)])]
        final_data[iid] = final_idx
    vocab = list(set(vocab))

###### save result CID-Dict and Vocab
with open(path + f"sequential-data-p5semid.json", "w") as f:
    json.dump([final_data, vocab], f)
