import numpy as np
import torch 
from torch import nn
from torch.nn import functional as F

class model(nn.Module):
    def __init__(self,inp_dim,interm,out_dim):
        super().__init__()
        self.layer1 = nn.Linear(inp_dim,interm)
        self.layer2 = nn.Linear(interm,out_dim)
    def forward(x):
        o = F.relu(self.layer1(x))
        return self.layer2(o)

if __name__=='__main__':
    m = model(784,200,10)
    print (m)

