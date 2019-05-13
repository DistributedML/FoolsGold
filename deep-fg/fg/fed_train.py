import argparse

from trainer import BaseTrainer
from trainer import FedTrainer
from option import Option
import pdb
import os
import utils 
import numpy as np
import pandas as pd
import math
import sys
import time
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

def get_model(option):
    arch = option.arch_type + str(option.arch_depth)

    if option.arch_type == "squeeze":
        # model = models.__dict__["squeezenet1_1"](num_classes=option.n_classes)
        model = models.__dict__["squeezenet1_1"](pretrained=option.pretrained)
        model.classifier[1] = nn.Conv2d(512, option.n_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = option.n_classes
        model = model.cuda()
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        return model

    if option.dataset == "vggface2":
        if option.pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=option.pretrained)
            if option.arch_type == "resnet":
                model.fc = nn.Linear(model.fc.in_features, option.n_classes, bias=True)
            elif option.arch_type == "vgg":
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, option.n_classes, bias=True)

            model = model.cuda()
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]().cuda()
    else:
        print("Invalid dataset")
        assert False
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
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

def get_loader(option, iid=[.0, .0]):
    client_loaders = []
    sybil_loaders = []
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
        train_df = df[df['train_flag'] == 0]
        val_df = df[df['train_flag'] == 1]
        new_train_df = pd.DataFrame()
        new_val_df = pd.DataFrame()
        for i in range(option.n_classes):
            new_val_df = new_val_df.append(val_df[val_df['idx'] == i])
            new_train_df = new_train_df.append(train_df[train_df['idx'] == i])
        train_df = new_train_df
        val_df = new_val_df

        num_clients = [1]*option.n_classes
        # Add honest clients
        for i in range(len(num_clients)):
            n_client_per_class = num_clients[i]
            for c in range(n_client_per_class):

                """
                N_iid / (N_s + N_iid) = x
                If iid=0, 100% single class + 0% Entire dataset P(y_i = i) = 1.0
                If iid=x, (100 - x)% single class + x% Entire dataset 
                If iid=100, 0% single class + 100% Entire dataset P(y_i = i) = 1/num_classes
                """
                if iid[0] == 1.0:
                    df = train_df.reset_index() # clients have iid data                
                elif iid[0] == 0.0:
                    df = train_df[train_df['idx'] == i].reset_index() # Clients have non-iid data
                else:
                    class_df = train_df[train_df['idx'] == i]
                    N_s = len(class_df)
                    N_iid = N_s * iid[0]/(1 - iid[0])
                    noniid_df = train_df

                    if len(noniid_df) < N_iid:
                        factor = math.ceil(N_iid / len(noniid_df)) - 1 # Factor is guaranteed to be >= 1
                        noniid_df = noniid_df.append([noniid_df]*factor, ignore_index=True)
                        noniid_df = noniid_df.sample(n=int(N_iid))
                    else:
                        noniid_df = noniid_df.sample(n=int(N_iid))
                    df = class_df.append(noniid_df, ignore_index=True).reset_index()

                clientset = VGG_Face2(df, datadir, train_transform)
                client_loader = torch.utils.data.DataLoader(clientset, batch_size=option.batch_size,
                                            num_workers=option.num_workers,
                                            shuffle=True, pin_memory=True)
                client_loaders.append(client_loader)
        for g in range(option.num_sybil_groups):
            source_class = g*2
            target_class = g*2 + 1
            # Add sybils
            for i in range(option.num_sybils):
                if iid[1] == 1.0:
                    df = train_df.copy() # Sybils have iid data
                    df.loc[df['idx'] == source_class, "idx"] = target_class
                    df = df.append(df[df['idx'] != target_class]) # Ensure iid
                    df = df.reset_index()
                elif iid[1] == 0.0:
                    df = train_df[train_df['idx'] == source_class].reset_index() # Sybils have non-iid data
                    df['idx'] = target_class
                else:
                    # Get all the 0's and corresponding amount of iid data
                    class_df = train_df[train_df['idx'] == source_class]
                    N_s = len(class_df)
                    N_iid = N_s * iid[1]/(1 - iid[1])
                    noniid_df = train_df

                    if len(noniid_df) < N_iid:
                        factor = math.ceil(N_iid / len(noniid_df)) - 1 # Factor is guaranteed to be >= 1
                        noniid_df = noniid_df.append([noniid_df]*factor, ignore_index=True)
                        noniid_df = noniid_df.sample(n=int(N_iid))
                    else:
                        noniid_df = noniid_df.sample(n=int(N_iid))
                    df = class_df.append(noniid_df, ignore_index=True)
                    df.loc[df['idx'] == source_class, "idx"] = target_class        
                    df = df.reset_index()

                clientset = VGG_Face2(df, datadir, train_transform)
                client_loader = torch.utils.data.DataLoader(clientset, batch_size=option.batch_size,
                                            num_workers=option.num_workers,
                                            shuffle=True, pin_memory=True)
                sybil_loaders.append(client_loader)
        
        train_df = train_df.reset_index()
        val_df = val_df.reset_index()

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

    return train_loader, val_loader, test_loader, client_loaders, sybil_loaders


def train(option, iid=[.0, .0]):
    model = get_model(option)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), option.lr,
        momentum=option.momentum, 
        weight_decay=option.weight_decay)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=option.lr)
    
    cudnn.benchmark = True
    # client loaders is the train loaders for every client
    train_loader, val_loader, test_loader, client_loaders, sybil_loaders = get_loader(option, iid)    
    start_time = time.time()
    trainer = FedTrainer(option, model, train_loader, val_loader, test_loader, optimizer, criterion, client_loaders, sybil_loaders, iidness=iid)
    trainer.train()
    end_time = time.time()
    print("Elapsed Time: {}".format(start_time - end_time))
    state = trainer.save_state()
    pdb.set_trace()
    return state


def main():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('conf_path', type=str, metavar='conf_path')
    args = parser.parse_args()

    option = Option(args.conf_path)
    train(option, iid=[1.0, 1.0])

if __name__ == "__main__":
    main()