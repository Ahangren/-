"""
用于处理数据样本的代码可能会变得混乱且难以维护;理想情况下，我们希望我们的数据集代码 与我们的模型训练代码解耦，以提高可读性和模块化。 PyTorch 提供了两个数据基元：它们允许您使用预加载的数据集以及您自己的数据。 存储样本及其相应的标签，并将 iterable 包装在 以便轻松访问样本。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
# 创建数据迭代器
training_data=datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data=datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure=plt.figure(figsize=(8,8))
cols,rows=3,3
for i in range(1,cols*rows+1):
    sample_idx=torch.randint(len(training_data),size=(1,)).item()
    img,label=training_data[sample_idx]
    figure.add_subplot(rows,cols,i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(img.squeeze(),cmap='gray')
plt.show()


# 采用自定义Dataset数据集
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self,annotations_file,img_dir,transform=None,target_transform=None):
        self.img_label=pd.read_csv(annotations_file)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform
        self.mast=1

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        img_path=os.path.join(self.img_dir,self.img_label.iloc[idx,0])
        image=read_image(img_path)
        label=self.img_label.iloc[idx,1]
        if self.transform:
            image=self.transform(image)
        if self.target_transform:
            label=self.target_transform(label)

        return image,label

train_dataloader=DataLoader(training_data,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=64,shuffle=True)

# 遍历dataloader
train_features,train_labels=next(iter(train_dataloader))
print(f'feature batch shape: {train_features.size()}')
print(f'labels batch shape: {train_labels.size()}')

img=train_features[0].squeeze()
label=train_labels[0]
plt.imshow(img,cmap='gray')
plt.show()
print(f'label: {label}')

