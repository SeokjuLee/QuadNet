# Seokju Lee 2017.12.27
"""
Load GTSRB dataset
img, label
"""
import os.path
import random
import glob
import math
from .listdataset import ListDataset
import torch
import pdb
import matplotlib.pyplot as plt
from torch.utils.serialization import load_lua
import numpy as np


def make_dataset(base, imfile, gtfile, split=100):
    '''
    Will make list of image path and label
    'img.png  /  label'
    '''
    images = []
    labels = []
    

    with open(imfile, 'r') as txtfile:
        for line in txtfile:
            elem = line.split('\n')
            images.append([os.path.join(base, elem[0])])
    with open(gtfile, 'r') as txtfile:
        for line in txtfile:
            elem = line.split('\n')
            labels.append([int(elem[0])])

    output = np.concatenate((np.array(images), np.array(labels)), axis=1).tolist()
    # pdb.set_trace()

    assert(len(output) > 0)
    random.shuffle(output)

    split_index = int(math.floor(len(output)*split/100))
    assert(split_index >= 0 and split_index <= len(output))
    # pdb.set_trace()
    return output[:split_index] if split_index < len(output) else output


def make_tempset(tempfile):
    '''
    Will make list of image path and label
    'img.png  /  label'
    '''
    output = []
    
    temp_list = sorted( os.listdir(tempfile) )
    for i in range(len(temp_list)):
        output.append([os.path.join(tempfile, temp_list[i]), str(i)])

    assert(len(output) > 0)
    random.shuffle(output)

    return output


def gtsrb_data(base, tr, tr_gt, te, te_gt, tp_tr, tp_te, transform=None, split=100, should_invert=False):
    train_list   = make_dataset(base, tr, tr_gt)
    test_list    = make_dataset(base, te, te_gt, split)
    temp_list_tr = make_tempset(tp_tr)
    temp_list_te = make_tempset(tp_te)
    # pdb.set_trace()

    train_dataset = ListDataset(train_list, temp_list_tr, transform, should_invert)
    test_dataset  = ListDataset(test_list,  temp_list_te, transform, should_invert)

    return train_dataset, test_dataset
