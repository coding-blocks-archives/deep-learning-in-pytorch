import torch
from torch import nn
from torch.nn import functional as F

def mse(yt,yp):
    return ((yp-yt)**2).mean()

def relu(x):
    idx = x<0
    x[idx]=0
    return x

def sigmoid(x):
    return x.exp()/ ( 1 + x.exp())


def softmax(x):
   return x.exp() / x.exp().sum(1)[:,None]

if __name__=='__main__':
   x = torch.randn(2,2)
   print (F.sigmoid(x))
   print (sigmoid(x))
   print (F.softmax(x))
   print (softmax(x))
