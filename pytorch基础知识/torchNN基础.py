from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import requests
import torch
import pickle
import gzip



DATA_PATH = Path('data')
PATH = DATA_PATH / 'mnist'
PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open('wb').write(content)

with gzip.open((PATH/FILENAME).as_posix(),'rb') as f:
    ((x_train,y_train),(x_valid,y_valid),_)=pickle.load(f,encoding='latin-1')

plt.imshow(x_train[0].reshape((28,28)),cmap='gray')

try:
    import google.colab
except ImportError:
    plt.show()
print(x_train.shape)

x_train,y_train,x_valid,y_valid=map(torch.tensor,x_train,y_train,x_valid,y_valid)

n,c=x_train.shape


from torch import nn
import math

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights=nn.Parameter(torch.randn(784,10)/math.sqrt(784))
        self.bias=nn.Parameter(torch.randn(10))

    def forward(self,xb):
        return torch.matmul(xb,self.weights)+self.bias


















