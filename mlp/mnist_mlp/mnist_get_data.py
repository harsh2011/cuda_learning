import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

import os


print('Using PyTorch version:', torch.__version__)

batch_size = 1

train_dataset = datasets.MNIST('./data', 
                               train=True, 
                               download=True, 
                               transform=transforms.ToTensor())

validation_dataset = datasets.MNIST('./data', 
                                    train=False, 
                                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False)

key = set()

for (X_train, y_train) in train_loader:
    if len(key) == 10:
        break
    
    y_train = y_train.numpy().flatten()
    X_train = X_train.view(-1, 28*28).numpy()

    if not y_train[0] in key:
        print(X_train.shape)
        np.save("images/"+str(y_train[0])+".npy",X_train)
        key.add(y_train[0])