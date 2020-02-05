import torch
import numpy as np

print ( 'Pytorch version --',torch.__version__)
print ( 'Pytorch cuda available --',torch.cuda.is_available() )

## From Numpy to torch tensors
x =  [1,2,3]
np_x = np.array(x)
tt_x = torch.tensor(x)

# from pytorch --> numpy
print (tt_x.numpy())

# Sum 
print (np_x.sum(), tt_x.sum())

# Getting scalar value from tensor
print (tt_x.sum().item())

# Mean
print (np_x.mean(), tt_x.float().mean()) # mean is cal on float tensors

# Matix Multiplication

t = torch.tensor([[1,2],[3,4]])

print (t@t)

# Taking Transpose 
print (t.T)





