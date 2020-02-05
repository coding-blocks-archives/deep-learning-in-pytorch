import torch

device = ('cuda:' + gpu if torch.cuda.is_available() else 'cpu')

x = torch.tensor([1,2,3]).to(device)
print (x)

# x = x.cuda()

print (x*x)

print (x.cpu())
