import torch

a=torch.randn(2,2,3,4)
b=torch.randn(1,1,3,4)
print((a+b).shape)