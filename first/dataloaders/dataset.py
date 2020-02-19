import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset
 
ds = pd.read_csv('../../data/mnist/mnist_train.csv')
print (ds.shape)
x = ds.values[:,1:]
y = ds.values[:,0]

class mnist_data(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __len__(self):
        return x.shape[0]
    def __getitem__(self,ix):
       return (x[ix],y[ix])


trn_data =  mnist_data(x[:50000],y[:50000])
#print (trn_data[0])

from torch.utils.data import DataLoader

trn_dl = DataLoader(trn_data,batch_size = 32,shuffle=True)

xb,yb = next(iter(trn_dl))
print (xb.shape,yb.shape)
#print (xb,yb)



