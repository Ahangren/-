import numpy
import pandas
import matplotlib
import torch


arr=torch.randn((1,255,255))
a=torch.randn(10)
a=a[:,None,None]
mask=(arr==a).to(dtype=torch.uint8)
print(mask)