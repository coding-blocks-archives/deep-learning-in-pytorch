import torch
import math

W = torch.randn(784,10)
print (W.requires_grad)  # by default tenors are fixed or no-gradients are calucated


# Method 1 
W1 = torch.randn(784,10,requires_grad=True) 
## --> any mathematetical operation with be mained in history for gradient calculation
L = 2*W1
print (L)


# Method 2
W2 = torch.tensor(784/10)/math.sqrt(784)
W2.requires_grad_()
print (W2.requires_grad)

# Calculate Gradients
W = torch.randn(5,5,requires_grad=True)
loss = (2*W).sum()
print ( loss)
loss.backward()
print (W.grad)


