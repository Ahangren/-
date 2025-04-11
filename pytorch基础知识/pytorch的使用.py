import torch
from torchvision.models import resnet18,ResNet18_Weights

model=resnet18(weights=ResNet18_Weights.DEFAULT)
# 创建数据
data=torch.rand(1,3,64,64)
label=torch.rand(1,1000)

# 前向传播
y_label=model(data)

# 创建优化器
optim=torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.9)

loss=(y_label-label).sum()

loss.backward()

optim.step()

print(loss)










