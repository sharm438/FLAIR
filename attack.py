# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:08:11 2021

@author: sharm438
"""

import torch
from copy import deepcopy
import pdb

def benign(device, param_list, cmax=0):

    return param_list

def full_trim(device, param_list, cmax=0):

    max_dim = torch.max(param_list, axis=0)[0]
    min_dim = torch.min(param_list, axis=0)[0]
    direction = torch.sign(torch.sum(param_list, axis=0)).to(device)
    directed_dim = (direction > 0) * min_dim + (direction < 0) * max_dim
    
    for i in range(cmax):
        random_12 = 1 + torch.rand(len(param_list[0])).to(device)
        param_list[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
           
    return param_list
    

