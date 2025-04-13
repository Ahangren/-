import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.nn.init as init
import sys
# from pytorch基础知识.图片和视频教学.TorchVision对象检测微调教程 import output

# 1数据加载
ds=datasets.load_iris(return_X_y=False)
print(f'数据集的特征属性名称列表：{ds.feature_names}')
print(f'数据集的类编名称列表{ds.target_names}')
in_features=len(ds.feature_names)
num_classes=len(ds.target_names)
X,Y=ds.data,ds.target

print(f'数据集的形状{type(X)}--{ds.feature_names}')

x_train,x_test,y_train,y_test=train_test_split(
    X,Y,test_size=0.2,random_state=14
)


class ClassifyNetwork(nn.Module):
    def __init__(self,in_features,num_classes):
        super(ClassifyNetwork,self).__init__()
        self.w1=nn.Parameter(torch.empty(in_features,8))
        self.w2=nn.Parameter(torch.empty(8,8))
        self.w3=nn.Parameter(torch.empty(8,num_classes))
        self.b1=nn.Parameter(torch.empty(8))
        self.b2=nn.Parameter(torch.empty(8))
        self.b3=nn.Parameter(torch.randn(3))

        for param in self.parameters():
            nn.init.normal_(param.data)

    def forward(self,x):
        # 输入层到隐藏层的全连接操作
        n1=torch.matmul(x,self.w1)+self.b1
        # n2=torch.matmul(n1,self.w2)+self.b2
        o1=nn.functional.relu(n1)
        n2=torch.matmul(o1,self.w2)+self.b2
        o2=nn.functional.relu6(n2)
        output=torch.matmul(o2,self.w3)+self.b3
        return output

def tt01():
    in_features=4
    net=ClassifyNetwork(in_features=in_features,num_classes=3)
    print('='*80)
    for _name,_param in net.named_parameters():
        print(_name,_param.requires_grad,_param)
    print('='*82)
    _x=torch.randn(5,in_features)
    _s=net(_x)
    print(_s)
    print(_s.shape)

from torch import optim
# optim=Optimizer()
net=ClassifyNetwork(in_features=4, num_classes=3)
loss_fn=nn.CrossEntropyLoss()
opt_fn=optim.SGD(net.parameters(),lr=1e-2,momentum=0.9)


if __name__ == '__main__':
    # net=ClassifyNetwork()
    # _x=torch.randn(5,4)
    # for _param in net.parameters():
    #     print(_param)
    tt01()
