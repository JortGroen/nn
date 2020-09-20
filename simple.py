#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:14:06 2020

@author: djoghurt
"""

# conda activate nn && tensorboard --logdir=D:/Jort/Documents/nn/runs

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
import loggernn

#create network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

writer = SummaryWriter('runs/mnist_simple_1')


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# hyperperameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 20

# load data
train_dataset = datasets.MNIST(root = 'dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root = 'dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

images, labels = next(iter(train_loader))
img_grid = make_grid(images)
#img_grid = images
#img_grid = img_grid.view(-1, img_grid.shape[2], img_grid.shape[0])
#img_grid = img_grid.view(img_grid.shape[1], img_grid.shape[2], img_grid.shape[0])
#plt.imshow(img_grid, cmap='winter')
writer.add_image('four test images', img_grid)

# Initialise network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# loss and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

t = time.time()
# train network
for epoch in range(num_epochs):
    
    print('epoch: ', epoch)
    
    for batchIdx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # reshape
        data = data.reshape(data.shape[0], -1)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        # backwared
        optimiser.zero_grad()
        loss.backward()
        
        # gradient decent
        optimiser.step()
        
    
    loggernn.tensorboard_update(writer, model, criterion, device, train_loader, test_loader, epoch)

print('elapsed time = ',time.time()-t)        
                        
loggernn.check_accuracy(train_loader, model, criterion, device)
loggernn.check_accuracy(test_loader, model, criterion, device)

writer.flush()
writer.close()
            
            