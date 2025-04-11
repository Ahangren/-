import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

# 修复1：transform.ToTensor 应该是 ToTensor()
transform = transforms.Compose([
    transforms.ToTensor(),  # 添加括号
    transforms.Normalize((0.5,), (0.5,))  # 修复2：元组末尾加逗号
])

# 加载数据集
trainset = torchvision.datasets.FashionMNIST(
    './data',
    download=True,
    train=True,
    transform=transform
)

testset = torchvision.datasets.FashionMNIST(
    './data',
    download=True,
    train=False,
    transform=transform
)

# 修复3：变量名拼写错误 (trstset -> testset)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=True
)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')  # 修复4：plt.imshow 不是 plt.show
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 修复5：所有训练代码必须放在 __main__ 中
if __name__ == '__main__':
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    writer = SummaryWriter('runs/fashion_mnist_experiment_1')

    # 测试数据加载
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # 创建图像网格
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)
    writer.add_image('four_fashion_mnist_images', img_grid)

    # 训练代码...
    # 添加你的训练循环在这里