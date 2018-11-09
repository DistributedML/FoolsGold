import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class MNISTCNNModel(nn.Module):
    # def __init__(self):
    #     super(MNISTCNNModel, self).__init__()
    #     self.layer1 = nn.Sequential(
    #         nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2))
    #     self.layer2 = nn.Sequential(
    #         nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2))
    #     self.drop_out = nn.Dropout()
    #     self.fc1 = nn.Linear(128, 128)
    #     self.fc2 = nn.Linear(128, 10)

    # def forward(self, x):
    #     out = self.layer1(x)
    #     out = self.layer2(out)
    #     out = out.reshape(out.size(0), -1)
    #     out = self.drop_out(out)
    #     out = self.fc1(out)
    #     out = self.fc2(out)
    #     return out
    
    # def reshape(self, flat_gradient):
    #     layers = []
    #     # ONE LAYER
    #     l1 = 16*1*5*5
    #     l2 = 16
    #     l3 = 32*16*5*5
    #     l4 = 32
    #     l5 = 128*128
    #     l6 = 128
    #     l7 = 10*128
    #     l8 = 10

    #     layers.append( torch.from_numpy( np.reshape(flat_gradient[0:l1], (16, 1, 5, 5))).type(torch.FloatTensor) )
    #     layers.append( torch.from_numpy( np.reshape(flat_gradient[l1:l1+l2], (16, ))).type(torch.FloatTensor) )
    #     layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2: l1+l2+l3], (32, 16, 5, 5))).type(torch.FloatTensor) )
    #     layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3 : l1+l2+l3+l4], (32, ))).type(torch.FloatTensor) )
    #     layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3+l4 : l1+l2+l3+l4+l5], (128, 128))).type(torch.FloatTensor) )
    #     layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3+l4+l5 : l1+l2+l3+l4+l5+l6], (128, ))).type(torch.FloatTensor) )
    #     layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3+l4+l5+l6 : l1+l2+l3+l4+l5+l6+l7], (10, 128))).type(torch.FloatTensor) )
    #     layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3+l4+l5+l6+l7 : l1+l2+l3+l4+l5+l6+l7+l8], (10, ))).type(torch.FloatTensor) )

    #     return layers
    def __init__(self):
        super(MNISTCNNModel, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 5, 1, 4), # output space (16, 16, 16)
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(2))
        # self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(16*16*16, 10)

    def forward(self, x):
        # x = x.view(x.shape[0], 28, 28)
        # x = x.unsqueeze(1)

        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        # out = self.drop_out(out)
        out = self.fc1(out)
        return out
    
    def reshape(self, flat_gradient):
        layers = []
        layers.append( torch.from_numpy(np.reshape(flat_gradient[0:400], (16, 1, 5, 5))).type(torch.FloatTensor) )
        layers.append( torch.from_numpy( np.reshape(flat_gradient[400:400+16], (16, ))).type(torch.FloatTensor) )
        layers.append( torch.from_numpy( np.reshape(flat_gradient[400+16: 400+16 + 10*4096], (10, 4096))).type(torch.FloatTensor))
        layers.append( torch.from_numpy( np.reshape(flat_gradient[400+16 + 10*4096: 400+16 + 10*4096 + 10], (10, ))).type(torch.FloatTensor))
        return layers