import random
import os

sequential_data = []
path = 'data/amazon/filtered/Yelp/lgcn/'
with open(os.path.join(path, 'sequential_data.txt'), 'r') as fd:
    for line in fd:
        sequential_data.append(line.rstrip('\n'))

# out.write(user + ' ' + ' '.join(items) + '\n')
train_set = {}
val_set = {}
test_set = {}
for line in sequential_data:
    user, items = line.strip().split(' ', 1)
    items = items.split(' ')
    item_len = len(items)
    train_set[user] = items[:-2]
    test_set[user] = items[-2:]

with open(os.path.join(path, 'train.txt'), 'w') as out:
    for user, items in train_set.items():
        out.write(user + ' ' + ' '.join(items) + '\n')


with open(os.path.join(path, 'test.txt'), 'w') as out:
    for user, items in test_set.items():
        out.write(user + ' ' + ' '.join(items) + '\n')