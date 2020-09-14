#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:14:06 2020

@author: djoghurt
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#create network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #print(x.shape)
        return x
    
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
        
# evaluation
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')
        
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            #x = x.reshape(x.shape[0], -1)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0) 
            
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()
    
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

# loss and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=learning_rate)

data, targets = next(iter(train_loader))
#targets = targets.view(1, -1)
print(targets.shape)
#data2 = data.reshape(data.shape[0], -1)
#data = data.squeeze(1)

# train network
for epoch in range(num_epochs):
    
    print('epoch: ', epoch)
    
    for batchIdx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        
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
    
#check_single(data, targets, model)
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
            
            

            
            