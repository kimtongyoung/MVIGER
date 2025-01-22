import random
import time
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.cluster import SpectralClustering
import numpy as np
from tqdm import tqdm
import json


# Sequential
def sequential_indexing(user_items):
    user2id = {}  # raw 2 uid
    item2id = {}  # raw 2 iid-old
    id2user = {}  # uid 2 raw
    id2item = {}  # iid-old 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    random_user_list = list(user_items.keys())
    random.shuffle(random_user_list)
    item_id = 1001
    # training item sequence only
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = []  # item id lists
        train_items = items[:-2]
        for item in train_items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    # assigning rest of items(val, test)
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = []  # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id - 1, item_id - 1, data_maps


def independent_indexing(user_items):
    user2id = {}  # raw 2 uid
    item2id = {}  # raw 2 iid-old
    id2user = {}  # uid 2 raw
    id2item = {}  # iid-old 2 raw
    user_id = 0
    item_id = 0
    final_data = {}
    user_list = list(user_items.keys())
    # random_user_list = list(user_items.keys())
    # random.shuffle(random_user_list)

    for user in user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = []  # item id lists
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


def random_indexing(user_items):
    user2id = {}  # raw 2 uid
    item2id = {}  # raw 2 iid-old
    id2user = {}  # uid 2 raw
    id2item = {}  # iid-old 2 raw
    user_id = 0
    item_id = 0
    final_data = {}
    random_user_list = list(user_items.keys())
    # random.shuffle(random_user_list)
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                item_id += 1
    item_id_list = [i + 1 for i in range(item_id - 1)]
    item2id = {}
    item_idx = 0
    random.shuffle(item_id_list)
    for user in random_user_list:
        items = user_items[user]
        iids = []  # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id_list[item_idx])
                id2item[str(item_id_list[item_idx])] = item
                item_idx += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id, item_idx, data_maps


def collaborative_indexing(user_items, cluster_num, token_size, last_token='sequential'):
    """
    Use collaborative indexing method to index the given user seuqnece dict.
    """
    user2id = {}  # raw 2 uid
    id2user = {}  # uid 2 raw
    user_id = 0
    final_data = {}
    random_user_list = list(user_items.keys())
    # random.shuffle(random_user_list)
    for user in random_user_list:
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1

    all_items = set()
    train_items = set()
    for user in user_items:
        all_items.update(set(user_items[user]))
        train_items.update(set(user_items[user][:-2]))

    # reindex all training items for calculating the adjacency matrix
    item2id = dict()
    id2item = dict()
    for item in train_items:
        item2id[item] = len(item2id)
        id2item[len(id2item)] = item

    # calculate the co-occurrence of items in the training data as an adjacency matrix
    adj_matrix = np.zeros((len(item2id), len(item2id)), dtype=np.float32)

    for user in user_items:
        interactions = user_items[user][:-2]
        for pairs in combinations(interactions, 2):
            adj_matrix[item2id[pairs[0]]][item2id[pairs[1]]] += 1
            adj_matrix[item2id[pairs[1]]][item2id[pairs[0]]] += 1

    # get the clustering results for the first layer
    begin_time = time.time()
    clustering = SpectralClustering(
        n_clusters=cluster_num,
        assign_labels="cluster_qr",
        random_state=0,
        affinity="precomputed",
    ).fit(adj_matrix)
    end_time = time.time()
    used_time = end_time - begin_time
    print("used time to compute it is {} seconds".format(used_time))
    labels = clustering.labels_.tolist()

    # count the clustering results
    grouping = defaultdict(list)
    for i in range(len(labels)):
        grouping[labels[i]].append((id2item[i], i))

    item_map = dict()
    index_now = 0

    # add current clustering information into the item indexing results.
    item_map, index_now = add_token_to_indexing(item_map, grouping, index_now, token_size)

    # add current clustering info into a queue for BFS
    queue = []
    for group in grouping:
        queue.append(grouping[group])

    # apply BFS to further use spectral clustering for large groups (> token_size)
    while queue:
        group_items = queue.pop(0)

        # if current group is small enough, add the last token to item indexing
        if len(group_items) <= token_size:
            item_list = [items[0] for items in group_items]
            if last_token == 'sequential':
                item_map = add_last_token_to_indexing_sequential(item_map, item_list, token_size)
            elif last_token == 'rid':
                item_map = add_last_token_to_indexing_random(item_map, item_list, token_size)
        else:
            # calculate the adjacency matrix for current group
            sub_adj_matrix = np.zeros((len(group_items), len(group_items)), dtype=np.float32)

            for i in range(len(group_items)):
                for j in range(i + 1, len(group_items)):
                    sub_adj_matrix[i][j] = adj_matrix[group_items[i][1]][group_items[j][1]]
                    sub_adj_matrix[j][i] = adj_matrix[group_items[j][1]][group_items[i][1]]

            # get the clustering results for current group
            clustering = SpectralClustering(
                n_clusters=cluster_num,
                assign_labels="cluster_qr",
                random_state=0,
                affinity="precomputed",
            ).fit(sub_adj_matrix)
            labels = clustering.labels_.tolist()

            # count current clustering results
            grouping = defaultdict(list)
            for i in range(len(labels)):
                grouping[labels[i]].append(group_items[i])

            # add current clustering information into the item indexing results.
            item_map, index_now = add_token_to_indexing(item_map, grouping, index_now, token_size)

            # push current clustering info into the queue
            for group in grouping:
                queue.append(grouping[group])

    # if some items are not in the training data, assign an index for them
    remaining_items = list(all_items - train_items)
    if len(remaining_items) > 0:
        if last_token == 'sequential':
            item_map = add_last_token_to_indexing_sequential(item_map, remaining_items, token_size)
        elif last_token == 'rid':
            item_map = add_last_token_to_indexing_random(item_map, remaining_items, token_size)

    reindex_user_sequence_dict = reindex(user_items, user2id, item_map)

    id2item = {}
    for item in item_map:
        if item not in id2item:
            iid = item_map[item]
            id2item[iid] = item

    data_maps = {
        'user2id': user2id,
        'item2id': item_map,
        'id2user': id2user,
        'id2item': id2item
    }
    return reindex_user_sequence_dict, len(reindex_user_sequence_dict), len(item_map), data_maps


def add_token_to_indexing(item_map, grouping, index_now, token_size):
    for group in grouping:
        index_now = index_now % token_size
        for (item, idx) in grouping[group]:
            if item not in item_map:
                item_map[item] = ''
            item_map[item] += f'{index_now}>'
        index_now += 1
    return item_map, index_now


def add_last_token_to_indexing_random(item_map, item_list, token_size):
    last_tokens = random.sample([i for i in range(token_size)], len(item_list))
    for i in range(len(item_list)):
        item = item_list[i]
        if item not in item_map:
            item_map[item] = ''
        item_map[item] += f'{last_tokens[i]}'
    return item_map


def add_last_token_to_indexing_sequential(item_map, item_list, token_size):
    for i in range(len(item_list)):
        item = item_list[i]
        if item not in item_map:
            item_map[item] = ''
        item_map[item] += f'{i}'
    return item_map


def get_dict_from_lines(lines):
    """
    Used to get user or item map from lines loaded from txt file.
    """
    index_map = dict()
    for line in lines:
        info = line.split(" ")
        index_map[info[0]] = info[1]
    return index_map


def generate_user_map(user_sequence_dict):
    """
    generate user map based on user sequence dict.
    """
    user_map = dict()
    for user in user_sequence_dict.keys():
        user_map[user] = str(len(user_map) + 1)
    return user_map


def reindex(user_sequence_dict, user_map, item_map):
    """
    reindex the given user sequence dict by given user map and item map
    """
    reindex_user_sequence_dict = dict()
    for user in user_sequence_dict:
        uid = user_map[user]
        items = user_sequence_dict[user]
        reindex_user_sequence_dict[uid] = [item_map[i] for i in items]

    return reindex_user_sequence_dict


def construct_user_sequence_dict(user_sequence):
    """
    Convert a list of string to a user sequence dict. user as key, item list as value.
    """

    user_seq_dict = dict()
    for line in user_sequence:
        user_seq = line.split(" ")
        user_seq_dict[user_seq[0]] = user_seq[1:]
    return user_seq_dict


def code_to_index(code_list):
    temp = ''
    for code in code_list:
        temp += str(code)
        temp += '>'
    return temp[:-1]


def residual_quantized_semantic_indexing(user_items, asin_code):
    user2id = {}  # raw 2 uid
    item2id = {}  # raw 2 iid-old
    id2user = {}  # uid 2 raw
    id2item = {}  # iid-old 2 raw
    user_id = 0
    final_data = {}
    random_user_list = list(user_items.keys())
    id_count = {}
    # for asin in result:
    #     id = result[asin]
    #     if id not in id_count:
    #         id_count[id] = 0
    #     result[asin] = id + f'>{id_count[id]}'
    #     id_count[id] += 1
    # return result
    # random.shuffle(random_user_list)
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = []  # item id lists
        for item in items:
            if item not in item2id:
                raw_id = code_to_index(asin_code[item])
                if raw_id not in id_count:
                    id_count[raw_id] = 0
                    curr_idx = id_count[raw_id]
                    item_id = raw_id + f'>{curr_idx}'
                    id_count[raw_id] += 1
                else:
                    curr_idx = id_count[raw_id]
                    item_id = raw_id + f'>{curr_idx}'
                    id_count[raw_id] += 1
                item2id[item] = item_id
                id2item[item_id] = item
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids

    print(f'total_items: {len(id2item)}')

    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id, len(item2id), data_maps
