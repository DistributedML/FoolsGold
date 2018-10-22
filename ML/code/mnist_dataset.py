from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import pdb
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class MNISTDataset(Dataset):
    def __init__(self, filename, root_dir, transform=None, is_train=True):
        self.filename = filename
        self.root_dir = root_dir
        self.transform = transform
        data = np.load(os.path.join(root_dir, filename + ".npy"))
        n, d = data.shape

        self.X = data[:, 0:d - 1]
        self.y = data[:, -1]
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = Image.fromarray(np.reshape(self.X[idx], (28, 28)))
        if self.transform:
            sample = self.transform(sample)
        
        return {'image': sample, 'label': self.y[idx]}
    
    def getData(self):
        return self.X, self.y