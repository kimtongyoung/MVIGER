import pdb

import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'Yelp']:
    dataset = dataloader.Loader(path="../../../data/amazon/filtered/" + world.dataset + '/lgcn')
    # dataset = dataloader.Loader(path="../../../data/amazon/filtered/" + world.dataset + '/lgcn-split')
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}
