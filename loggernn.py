# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:17:22 2020

@author: Jort
"""
import torch
import torch.nn as nn

def tensorboard_update(writer, model, criterion, device, train_loader, test_loader, epoch):    
    loss, accuracy = check_accuracy(train_loader, model, criterion, device)
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)
    
    loss, accuracy = check_accuracy(test_loader, model, criterion, device)    
    writer.add_scalar('Loss/test', loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    
    
# evaluation
def check_accuracy(loader, model, criterion, device):
    # if loader.dataset.train:
    #     print('Checking accuracy on training data')
    # else:
    #     print('Checking accuracy on test data')
        
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            #x = x.reshape(x.shape[0], -1)
            x = x.squeeze(1)
            
            scores = model(x)
            loss = criterion(scores, y)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0) 
        accuracy = float(num_correct)/float(num_samples)*100    
        #print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
        model.train()
        return loss, accuracy