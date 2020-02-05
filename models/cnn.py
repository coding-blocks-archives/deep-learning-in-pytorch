import numpy as np
import torch 
from torch import nn
from torch.nn import functional as F

def cnn(nn.Module):
    def __init__(self,inp,ch1,out):
        super().__init__()
        self.conv1 = nn.Conv2d(inp,ch1,kernal=3)
        self.conv2 = nn.Conv2d(ch1,out,kernal=3)
    def forward(self,x):
        o = self.conv1(x)
        

