# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:56:31 2020

@author: Jort
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import loggernn
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter('runs/Bidirectional_LSTM')

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 100

class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        
        return out
    
# load data
train_dataset = datasets.MNIST(root = 'dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root = 'dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
                         
# Initialise network
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# loss and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

t = time.time()
# train network
for epoch in range(num_epochs):
    
    print('epoch: ', epoch)
    
    for batchIdx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)
               
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

writer.flush()
writer.close()
