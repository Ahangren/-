import torch
import torchvision
import torchvision.transforms as transforms

transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

batch_size=4

trainset=torchvision.datasets.CIFAR10(root='./data',train=True,transform=transforms,download=True)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)

testset=torchvision.datasets.CIFAR10(root='./data',train=False,transform=transforms,download=True)
testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 展示一些图片
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img=img/2*0.5
    img=np.array(img)
    img=np.transpose(img,(1,2,0))
    plt.imshow(img)
    plt.show()


dataiter=iter(trainloader)
images,labels=next(dataiter)
imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]}' for j in range(batch_size)))


import torch.nn as nn
import torch.nn.functional as F

# 定义卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=torch.flatten(x)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
net=Net()











