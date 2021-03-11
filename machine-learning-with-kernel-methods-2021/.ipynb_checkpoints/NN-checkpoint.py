import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import requests
import os


data_path = 'C:/Users/art-9/Desktop/ENS Paris-Sclay/M2/S2/Représentation Parcimonieuse/data challenge/'

X_train = pd.read_csv(data_path + 'train_X.csv')
Y_train = pd.read_csv(data_path + 'train_Y.csv')
#X_test = pd.read_csv(data_path + 'test_X.csv')

## Mise en place des données

Datasets_X = []
Datasets_Y = []

To_train = []
Last_obs = []

Last_obs_mean = np.zeros((5,192))
To_test = torch.zeros((5,192,2))
To10 = np.arange(0,10)

for i in range(5):
    Datasets_X.append(X_train[X_train['DATASET']==i].drop(columns = ['DATASET','ID'], axis = 0))
    Datasets_Y.append(Y_train[Y_train['DATASET']==i].drop(columns = ['DATASET','ID','VARIANCE'], axis = 0))

    Last_obsi = Datasets_X[i][Datasets_X[i]['TIME']==9]
    Last_obs.append(Last_obsi[Last_obsi['MODEL']==0]['VALUE'].to_numpy())

    for j in range(192):
        Set = np.arange(16*j,(1+j)*16)
        Last_obs_mean[i,j] = Last_obs[i][16*j:(1+j)*16].mean()

        z = Datasets_X[i][Datasets_X[i]['TIME']==10]
        z = z[z['POSITION']==j]['VALUE'].to_numpy()

        if i>0: # On créer le x_train qui ne comporte pas le dataset 0
            To_train.append((np.array([Last_obs_mean[i,j],z.mean()]),np.array([Datasets_Y[i]['MEAN'].to_numpy()[j]])))

        To_test[i][j] = torch.tensor(np.array([Last_obs_mean[i,j],z.mean()]))

## Creation du model

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        size = 500
        self.l1 = nn.Linear(2, size)
        self.l2 = nn.Linear(size, size)
        self.l3 = nn.Linear(size, size)
        self.l4 = nn.Linear(size, 1)
    def forward(self, inputs):

        out = torch.relu(self.l1(inputs))
        out = torch.relu(self.l2(out))
        out = torch.relu(self.l3(out))
        outputs = self.l4(out)

        return outputs

model = Model()
num_epochs = 215
batch_size = 16

criterion =  nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(num_epochs, batch_size, criterion, optimizer, model, train_set):
    train_error = []
    train_loader = DataLoader(train_set, batch_size, shuffle=False)
    model.train()

    for epoch in range(num_epochs):
        if epoch == 200:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        epoch_average_loss = 0.0

        for (x, y) in train_loader:
            y_pre = model(x.float())
            loss = criterion(y_pre, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_average_loss += loss.item() * batch_size / len(train_set)
        train_error.append(epoch_average_loss)

        if epoch %5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, epoch_average_loss))
    return train_error

train_error = train(num_epochs, batch_size, criterion, optimizer, model, To_train)

## Test sur le dataset 0

Err_obs = np.sum((Last_obs_mean[0]-Datasets_Y[0]['MEAN'].to_numpy())**2)
Err_NN = np.sum((model(To_test[0]).detach().numpy().reshape((192,))-Datasets_Y[0]['MEAN'].to_numpy())**2)

print('')
print('Erreur dernière observation : ' , Err_obs)
print('Erreur prédiction du NN :' , Err_NN)

Var = np.zeros(192)
for i in range(1,5):
    Var += (model(To_test[i]).detach().numpy().reshape((192,))-Datasets_Y[i]['MEAN'].to_numpy())**2
Var *= 5

R2 = np.sum((model(To_test[0]).detach().numpy().reshape((192,))-Datasets_Y[0]['MEAN'].to_numpy())**2)/np.sum(Datasets_Y[0]['MEAN'].to_numpy()**2)

Reliability = np.sqrt(((model(To_test[0]).detach().numpy().reshape((192,))-Datasets_Y[0]['MEAN'].to_numpy())**2/Var).mean())

print('-log(R2) : ' ,-np.log(R2))
print('-log(Reliability) : ' , -np.log(Reliability))

Score = -np.log(R2) - abs(np.log(Reliability))
print('Score : ' , Score)

























































