import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from twisted.conch.insults.text import flatten

# 配置device
device=torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(f'using{device} device')

# 创建网络结构
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self,x):
        x=self.fatten(x)
        x=self.linear_relu_stack(x)
        return x

# 实例化模型
model=NeuralNetwork().to(device)
# print(model)

# 构建数据
X=torch.rand(1,28,28,device=device)
logits=model(X)
print("模型运行完的数据：",logits)

# 因为是分类任务，所以使用softmax函数
pred_probab=nn.Softmax(dim=1)(logits)
print(pred_probab)
# 取出概率最大的值就是预测值
y_pred=pred_probab.argmax(1)
print(f'模型预测为：{y_pred}')

# 分解 FashionMNIST 模型中的层
input_image=torch.rand(3,28,28)

print(input_image)
# 展平
flatten=nn.Flatten()
flat_image=flatten(input_image)
print(flat_image.size())
# 线性层
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
# nn.ReLU 系列
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")


# nn.Sequential 是有序的 模块的容器。数据按照定义的顺序通过所有模块传递。您可以使用 顺序容器将 .seq_modules
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# 打印模型参数
for name,param in model.named_parameters():
    print(f'Layer: {name}| size: {param.size()}|values: {param[:2]}\n')





















