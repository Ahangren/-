import torch
import torch.nn as nn

"""
卷积神经网络包括：池化层，卷积层，全连接层，激活函数，反向传播
关键字：输入通道数，输出通道数，卷积核，池化，全连接层，激活函数
输出大小：（输入大小-卷积核大小+填充大小）/步长+1
"""

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        # 卷积操作，输入通道数为1，输出通道数为6，卷积核大小5*5
        self.conv1=nn.Conv2d(1,6,5)
        # 卷积操作，输入通道数为6，输出通道数为16，卷积核大小5*5
        self.conv2=nn.Conv2d(6,16,5)
        # 定义第一个池化层，使用最大值池化
        self.pool1=nn.MaxPool2d(2,2)
        # 定义第一个全连接层，输入参数为16*5*5，输出为120
        self.fc1=nn.Linear(16*5*5,120)
        # 定义第二个全连接层，输入参数为120，输出参数为84
        self.fc2=nn.Linear(120,84)
        # 定义输出层，输入参数为84，输出10个类别
        self.output=nn.Linear(84,10)

    def forward(self,x):
        # 第一层卷积+池化+激活函数
        x=self.conv1(x)
        x=nn.ReLU(x)
        x=self.pool1(x)
        # 第二层，卷积+激活函数+池化
        x=self.pool1(nn.ReLU(self.conv2(x)))
        # 将多维张量展为1维，方便进入全连接层
        x=x.view(-1,16*5*5)
        # 第一个全连接层
        x=self.fc1(x)
        x=nn.ReLU(x)
        # 第二个全连接层
        x=nn.ReLU(self.fc2(x))
        # 输出
        x=self.output(x)
        return x

def MKernel():
    in_channels = 5  # 输入通道数量
    out_channels = 10  # 输出通道数量
    width = 100  # 每个输入通道上的卷积尺寸的宽
    heigth = 100  # 每个输入通道上的卷积尺寸的高
    kernel_size = 3  # 每个输入通道上的卷积尺寸
    batch_size = 1  # 批数量

    input = torch.randn(batch_size, in_channels, width, heigth)
    conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

    out_put = conv_layer(input)

    print(input.shape)
    print(out_put.shape)
    print(conv_layer.weight.shape)

"""
输入维度：（batch_size，c_in，h_in，w_in）
卷积核维度：（h_k，w_k，c_k）
输出维度：（batch_size，c_out，h_out,w_out）
c_out=c_k
h_out=[h_in+2P-(k-1)-1]/s+1
w_out=[w_in+2P-(k-1)-1]/s+1

"""
"""
池化层计算：

"""

if __name__ == '__main__':
    MKernel()