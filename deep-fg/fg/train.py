import argparse

from trainer import BaseTrainer
from trainer import FedTrainer

from option import Option
import pdb
import os
import utils 
import numpy as np
import pandas as pd
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


def get_model(option):
    arch = option.arch_type + str(option.arch_depth)
    if option.dataset == "vggface2":
        if option.pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=option.pretrained).cuda()
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]().cuda()
    else:
        print("Invalid dataset")
        assert False
    return model

def get_train_val_sampler(option, trainset, valid_size=0.3):
    num_train = len(trainset)
    split = int(np.floor(valid_size * num_train))
    idx_path = os.path.join(option.data_path, "indices.npy")
    if os.path.isfile(idx_path):
        indices = np.load(idx_path)
    else:            
        indices = list(range(num_train))         
        np.random.shuffle(indices)
        np.save(idx_path, indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)
    return train_sampler, val_sampler

def get_loader(option):
    if option.dataset == "vggface2":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        datadir = os.path.join(option.data_path, "data")
        train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        df_path = os.path.join(option.data_path, "top10_files.csv")
        df = pd.read_csv(df_path)
        train_df = df[df['train_flag'] == 0].reset_index()
        val_df = df[df['train_flag'] == 1].reset_index()

        trainset = VGG_Face2(train_df, datadir, train_transform)  
        valset = VGG_Face2(val_df, datadir, test_transform)       
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=option.batch_size,
                                          num_workers=option.num_workers,
                                          shuffle=True,
                                          pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=option.batch_size,
                                          num_workers=option.num_workers,
                                          shuffle=True,
                                          pin_memory=True)
        # For now, just use train/val for poisoning experiments
        test_loader = val_loader
    else: 
        print("Invalid dataset")
        assert False

    return train_loader, val_loader, test_loader

def train(option):
    model = get_model(option)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), option.lr,
        momentum=option.momentum, 
        weight_decay=option.weight_decay)
    
    cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_loader(option)    

    trainer = FedTrainer(option, model, train_loader, val_loader, test_loader, optimizer, criterion)
    trainer.train()
    # trainer.validate()
    pdb.set_trace()


def main():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('conf_path', type=str, metavar='conf_path')
    args = parser.parse_args()

    option = Option(args.conf_path)

    train(option)

if __name__ == "__main__":
    main()