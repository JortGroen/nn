#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:14:06 2020

@author: djoghurt
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""

# conda activate nn && tensorboard --logdir=D:/Jort/Documents/nn/runs

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import loggernn
from torch.utils.tensorboard import SummaryWriter

    
class CNN(nn.Module):
    def __init__(self, in_shape, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))       # convolution
        x = F.max_pool2d(x, 2)          # subsampling
        x = F.relu(self.conv2(x))       # convolution
        x = F.max_pool2d(x, 3)          # subsampling
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))         # fully connected layer
        x = F.relu(self.fc2(x))         # fully connected layer
        x = self.fc3(x)
        #print(x.shape)
        return x
        
    
def check_single(x,y, model):
    print('checking data')
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        x = x.to(device=device)
        y = y.to(device=device)
        #x = x.reshape(x.shape[0], -1)
        scores = model(x)
        _,predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0) 
        
    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()
    

writer = SummaryWriter('runs/mnist_cnn_adam')
    
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# hyperperameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 500

# load data
train_dataset = datasets.MNIST(root = 'dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root = 'dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialise network
#model = NN(input_size=input_size, num_classes=num_classes).to(device)
model = CNN(in_shape=[batch_size, 1, 28, 28], num_classes=num_classes)
model.to(device)

# loss and optimiser
criterion = nn.CrossEntropyLoss()
# optimiser = optim.SGD(model.parameters(), lr=learning_rate)
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# data1, targets = next(iter(train_loader))
# #targets = targets.view(1, -1)
# print(targets.shape)
# model(data1)
# print(model(data1).shape)

#data2 = data.reshape(data.shape[0], -1)
#data = data.squeeze(1)

# train network
for epoch in range(num_epochs):
    
    print('epoch: ', epoch)
    
    for batchIdx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        # reshape
        #data = data.reshape(data.shape[0], -1)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        # backwared
        optimiser.zero_grad()
        loss.backward()
        
        # gradient decent(data, model)
        optimiser.step()
        
    loggernn.tensorboard_update(writer, model, criterion, device, train_loader, test_loader, epoch)
    

            
loggernn.check_accuracy(train_loader, model, criterion, device)
loggernn.check_accuracy(test_loader, model, criterion, device)

writer.flush()
writer.close()            

            
            