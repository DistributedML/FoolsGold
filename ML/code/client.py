import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import numpy as np
import pdb

class Client():
    def __init__(self, dataset_name, filename, batch_size, model, dataset, transform):
        # initializes dataset
        self.batch_size=batch_size
        self.trainset = dataset(filename, "data/" + dataset_name, is_train=True, transform=transform)
        self.testset = dataset("mnist_test", "data/" + dataset_name, is_train=False, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.full_loader = torch.utils.data.DataLoader(self.trainset, batch_size=len(self.trainset), shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=len(self.testset), shuffle=False)

        self.model = model

        ### Tunables ###
        # self.criterion = nn.MultiLabelMarginLoss()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5, weight_decay=0.001) # mnist_softmax
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.001) # lfw_cnn
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.001) # lfw_softmax
        self.aggregatedGradients = []
        self.loss = 0.0

    # TODO:: Get noise for diff priv
    def getGrad(self):
        for i, data in enumerate(self.trainloader, 0):
            # get the inputs
            inputs = data['image'].float()
            labels = data['label'].long()

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            # nn.utils.clip_grad_norm(self.model.parameters(), 100)
            self.loss = loss.item()

            # get gradients into layers
            layers = np.zeros(0)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # print(param.grad.shape)
                    layers = np.concatenate((layers, param.grad.numpy().flatten()), axis=None)
            return layers

    def simpleStep(self, gradient):
        layers = self.model.reshape(gradient)
        # Manually updates parameter gradients
        layer = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = layers[layer]
                layer += 1

        # Step in direction of parameter gradients
        self.optimizer.step()

    
    # Called when the aggregator shares the updated model
    def updateModel(self, modelWeights):
        
        layers = self.model.reshape(modelWeights)
        layer = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = layers[layer]
                layer += 1

    def getModelWeights(self):
        layers = np.zeros(0)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                layers = np.concatenate((layers, param.data.numpy().flatten()), axis=None)
        return layers

    def getLoss(self):
        return self.loss
    
    def getTrainErr(self):
        for i, data in enumerate(self.full_loader, 0):
            # get the inputs
            inputs = data['image'].float()
            labels = data['label'].long()
            inputs, labels = Variable(inputs), Variable(labels)
            out = self.model(inputs)
            pred = np.argmax(out.detach().numpy(), axis=1)
        return 1 - accuracy_score(pred, labels)

    def getTestErr(self):
        for i, data in enumerate(self.testloader, 0):
            # get the inputs
            inputs = data['image'].float()
            labels = data['label'].long()
            inputs, labels = Variable(inputs), Variable(labels)
            out = self.model(inputs)
            pred = np.argmax(out.detach().numpy(), axis=1)
        return 1 - accuracy_score(pred, labels)
