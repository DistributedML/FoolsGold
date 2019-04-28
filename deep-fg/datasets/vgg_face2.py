import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

import pdb

# Pass in dataframe
class VGG_Face2(Dataset):
    def __init__(self, df, datadir, transform=None):
        self.datadir = datadir
        self.transform = transform
        self.df = df
        
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_name = self.df['file'][idx]
        class_num = self.df['Class_ID'][idx]
        img_path = os.path.join(self.datadir, class_num, file_name)
        X = Image.open(img_path)
        y = torch.tensor(int(self.df['idx'][idx]))
        if self.transform:
            X = self.transform(X)
        return X, y