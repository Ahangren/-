import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib.font_manager import weight_dict
from openpyxl.styles.builtins import output

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
# 创建损失函数以及优化器
import torch.optim as optim
criterion=nn.CrossEntropyLoss()
optim=optim.SGD(Net.parameters(),lr=1e-2,momentum=0.9)
if __name__ == '__main__':

    for epoch in range(2):
        running_loss=0.0
        for i,data in enumerate(trainloader,0):
            inputs,labels=data
            optim.zero_grad()
            output=net(inputs)
            loss=criterion(output,labels)
            loss.backward()
            optim.step()

            running_loss+=loss.item()
            if i%2000==1999:
               print(f'{epoch+1},{i+1:5d} loss:{running_loss/2000:.3f}')
               running_loss=0.0
    print('Finished Training')


    # 保存模型
    PATH='./cifar_net.path'
    torch.save(net.state_dict(),PATH)


    # 测试一下我们训练的模型
    dataiter=iter(testloader)
    images,label=next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    print(f'GtoudTruth :',' '.join(f'{classes[label[j]]}' for j in range(4)))

    net=Net()

    net.load_state_dict(torch.load(PATH,weights_only=True))

    outputs=net(images)

    _,predicted=torch.max(outputs,1)
    print('Predicted:', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

    correct=0
    total=0
    with torch.no_grad():
        for data in testloader:
            image,label=data
            output=net(image)
            _,predict=torch.max(output,1)
            total+=label.size(0)
            correct+=(predict==label).sum().item()
    print(f"这给模型在一万个测试数据中的准确率为：{100*correct//total:.2f}%")


    # 看看有哪些类表现的比较好，有那些类表现的不好
    correct_pred={classname :0 for classname in classes}
    total_pred={classname:0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images,labels=data
            output=net(images)
            _,predict=torch.max(output,1)
            for label in labels:
                if predict==label:
                    correct_pred[classes[label]]+=1
                total_pred[classes[label]]+=1
    for classname,correct_count in correct_pred.items():
        accuracy=100*float(correct_count)/total_pred[classname]
        print(f"{classname}的准确率为{accuracy:.1f}")








