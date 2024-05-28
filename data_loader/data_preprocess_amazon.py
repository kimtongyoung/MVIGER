import re
from collections import defaultdict
import pandas as pd
import gzip
import json
import os
import random
import pickle
import torch
import numpy as np
import tqdm

seed = 2024
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# load data
def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        return json.load(f)


def ReadLineFromFile(path):
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in load_json(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def Amazon(dataset_name, rating_score):
    datas = []
    data_file = f'data/amazon/raw/{dataset_name}/reviews_{dataset_name}.json.gz'

    for inter in parse(data_file):
        if float(inter['overall']) <= rating_score:
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        datas.append((user, item, int(time)))
    return datas


def Amazon_meta(dataset_name, data_maps):
    datas = {}
    meta_file = f'data/{dataset_name}/meta_{dataset_name}.json.gz'
    item_asins = list(data_maps['item2id'].keys())
    for info in parse(meta_file):
        if info['asin'] not in item_asins:
            continue
        datas[info['asin']] = info
    return datas


def Yelp(dataset_name, date_min, date_max, rating_score):
    datas = []
    data_flie = f'data/{dataset_name}/yelp_academic_dataset_review.json'
    lines = open(data_flie, encoding='utf-8').readlines()
    for line in tqdm(lines):
        review = json.loads(line.strip())
        user = review['user_id']
        item = review['business_id']
        rating = review['stars']
        date = review['date']
        if date < date_min or date > date_max or float(rating) <= rating_score:
            continue
        time = date.replace('-', '').replace(':', '').replace(' ', '')
        datas.append((user, item, int(time)))
    return datas


def Yelp_meta(dataset_name, datamaps):
    meta_infos = {}
    meta_file = f'data/{dataset_name}/yelp_academic_dataset_business.json'
    item_ids = list(datamaps['item2id'].keys())
    lines = open(meta_file, encoding='utf-8').readlines()
    for line in tqdm(lines):
        info = json.loads(line)
        if info['business_id'] not in item_ids:
            continue
        meta_infos[info['business_id']] = info
    return meta_infos

def add_comma(num):
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num) - i - 1) % 3 == 0:
            res_num += ','
    return res_num[:-1]

def independent_indexing(user_items):
    user2id = {}
    item2id = {}
    id2user = {}
    id2item = {}
    user_id = 0
    item_id = 0
    final_data = {}
    user_list = list(user_items.keys())
    for user in user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = []
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    print(f'total_items: {item_id}')
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id, item_id, data_maps


def get_interaction(datas):
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq


def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True


def filter_Kcore(user_items, user_core, item_core):
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core:
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items


def main(data_name, rating_score, user_core, item_core, index_type):
    rating_score = rating_score
    user_core = user_core
    item_core = item_core
    if data_name == 'Yelp':
        date_max = '2019-12-31 00:00:00'
        date_min = '2019-01-01 00:00:00'
        datas = Yelp(data_name, date_min, date_max, rating_score=rating_score)
    else:
        datas = Amazon(data_name, rating_score=rating_score)

    user_items = get_interaction(datas)
    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    user_items, user_num, item_num, data_maps = independent_indexing(user_items)
    user_count, item_count, _ = check_Kcore(user_items, user_core=user_core, item_core=item_core)
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)

    print('Begin extracting meta infos...')
    if data_name == 'Yelp':
        meta_infos = Yelp_meta(data_name, data_maps)
    else:
        meta_infos = Amazon_meta(data_name, data_maps)

    print(f'{data_name} & user: {add_comma(user_num)}& item: {add_comma(item_num)} & user_avg: {user_avg:.1f}'
          f'& item_avg: {item_avg:.1f}& interactions: {add_comma(interact_num)}& '
          f'sparsity: {sparsity:.2f}  & meta_infos: {add_comma(len(meta_infos))}')

    # -------------- Save Data ---------------
    data_file = f'data/{data_name}/{index_type}/sequential_data.txt'
    meta_file = f'data/{data_name}/{index_type}/meta_data.json'
    datamaps_file = f'data/{data_name}/{index_type}/datamaps.json'

    with open(data_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')

    json_str = json.dumps(meta_infos)
    with open(meta_file, 'w') as out:
        out.write(json_str)

    json_str = json.dumps(data_maps)
    with open(datamaps_file, 'w') as out:
        out.write(json_str)


###############################################################################
# main('Yelp', 0, 5, 5, 'iid')
# main('Sports_and_Outdoors', 0, 5, 5, 'iid')
# main('Toys_and_Games', 0, 5, 5, 'iid')
###############################################################################
