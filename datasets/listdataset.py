# Seokju Lee 2018.03.29
"""
Load siamese list
"""
import torch.utils.data as data
import os
import os.path
from scipy.ndimage import imread
import numpy as np
import pdb
import matplotlib.pyplot as plt
from torch.utils.serialization import load_lua
import torch
from scipy import ndimage, misc
from PIL import Image
import random


class ListDataset(data.Dataset):
    def __init__(self, path_list, temp_list, transform=None, should_invert=False):
        self.path_list = path_list
        self.temp_list = temp_list
        self.transform = transform
        self.should_invert = should_invert
        # pdb.set_trace()

    def __getitem__(self, index):
        RA_list = self.path_list[index]
   
        # Randomly pick Real-B
        while True:
            #keep looping till the different class real image is found
            RB_list = random.choice(self.path_list)
            if RB_list[1] != RA_list[1]:
                break

        # Pick Temp-A
        while True:
            #keep looping till the Temp-A is found
            TA_list = random.choice(self.temp_list)
            if TA_list[1] == RA_list[1]:
                break

        # Pick Temp-B
        while True:
            #keep looping till the Temp-B is found
            TB_list = random.choice(self.temp_list)
            if TB_list[1] == RB_list[1]:
                break

        RA = Image.open(RA_list[0])
        RB = Image.open(RB_list[0])
        TA = Image.open(TA_list[0])
        TB = Image.open(TB_list[0])
        
        if self.should_invert:
            RA = PIL.ImageOps.invert(RA)
            RB = PIL.ImageOps.invert(RB)
            TA = PIL.ImageOps.invert(TA)
            TB = PIL.ImageOps.invert(TB)

        if self.transform is not None:
            RA = self.transform(RA)
            RB = self.transform(RB)
            TA = self.transform(TA)
            TB = self.transform(TB)
        # pdb.set_trace()

        return RA, RB, TA, TB, \
               torch.from_numpy(np.array( [float(RA_list[1])] )), \
               torch.from_numpy(np.array( [float(RB_list[1])] ))
                


    def __len__(self):
        return len(self.path_list)