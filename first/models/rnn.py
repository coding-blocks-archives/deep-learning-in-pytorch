import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class rnn(nn.Module):
    def __init__(self,vs,emb_dim,nl,hid,out):
        super().__init__()
        self.hid = hid
        self.nl = nl
        self.emb = nn.Embedding(vs,emb_dim)
        self.rnn = nn.RNN(emb_dim,hid,batch_first=True) 
        self.out = nn.Linear(hid,out)
    def forward(self,x):
       x = self.emb(x)
       h = torch.zeros( (self.nl,x.shape[0],self.hid),requires_grad=True)  
       o,h =  self.rnn( x,h)
       print (o.shape)
       return self.out(o[:,-1])

if __name__=='__main__':
    m = rnn(10,100,1,128,10)
    print (m)
    x = torch.randint(0,10,(2,5))

    print (m(x))


