import argparse

from trainer import BaseTrainer
from trainer import FedTrainer
from option import Option
import pdb
import os
import utils 
import numpy as np
import pandas as pd
import time
import sys
sys.path.append("../")

from datasets.vgg_face2 import VGG_Face2

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler

from models.lenet import LeNet
from fed_train import train

"""
Experiment to vary iid-ness of client and sybils using FG
Saves Val Accuracy, Val Poisoning Rate, Memory, and WVHistory
"""
def main():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('conf_path', type=str, metavar='conf_path')
    args = parser.parse_args()
    
    option = Option(args.conf_path)
    grid = np.zeros((5,5))
    for i, client_iid in enumerate([.0, .25, .5, .75, 1.0]):
        for j, sybil_iid in enumerate([.0, .25, .5, .75, 1.0]):
            end = time.time()
            state = train(option, [client_iid, sybil_iid])
            runtime = time.time() - end 
            print("Finished client iid: {} sybil iid: {} in {}".format(i, j, runtime))
            grid[i][j] = state['attack_rate']
    pdb.set_trace()
if __name__ == "__main__":
    main()