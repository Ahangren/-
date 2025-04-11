
import string
from turtledemo.penrose import start

import unicodedata

from io import open
import glob
import os
import time

import torch
from openpyxl.styles.builtins import output
from torch.utils.data import Dataset
from xarray.plot.utils import label_from_attrs

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'using device: {torch.get_default_device()}')

allowed_characters = string.ascii_letters + " .,;'" + "_"
n_letters = len(allowed_characters)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )


print(f"converting 'Ślusàrski' to {unicodeToAscii('Ślusàrski')}")

char_to_index = {char: idx for idx, char in enumerate(allowed_characters)}
def letterToIdex(letter):
    return char_to_index.get(letter, char_to_index['_'])


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIdex(letter)] = 1
    return tensor


print(f"这个单词的编码： {lineToTensor('letter')}")


class NamesDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_time = time.localtime
        labels_set = set()

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        text_files = glob.glob(os.path.join(data_dir, '*.txt'))
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]
            labels_set.add(label)
            lines = open(filename, encoding='utf-8').read().strip().split('\n')

            for name in lines:
                self.data.append(name)
                self.data_tensors.append(lineToTensor(name))
                self.labels.append(label)
        self.labels_uniq = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]  # 形状已经是 (1,)

        return label_tensor, data_tensor, data_label, data_item


alldata = NamesDataset('data1/data/names')
print(f'loaded {len(alldata)} items of data')
print(f'example={alldata[0]}')

train_set,test_set=torch.utils.data.random_split(alldata,[.85,.15],
                                                 generator=torch.Generator(device=device).manual_seed(2024))
print(f'train examples = {len(train_set)}, validation examples = {len(test_set)}')

import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=False)  # 使用seq_len first
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x形状: (seq_len, batch=1, input_size)
        rnn_out, _ = self.rnn(x)
        # 取最后一个时间步的输出
        last_out = rnn_out[-1]  # (1, hidden_size)
        output = self.h2o(last_out)
        return output
# 创建一个58个节点，128个隐藏节点和18给输出
n_hidden=128
rnn=CharRNN(n_letters,n_hidden,len(alldata.labels_uniq))
print(rnn)


# 训练
import random
import numpy as np

criterion = nn.CrossEntropyLoss()


def train(rnn, training_data, n_epoch=10, n_batch_size=64, report_every=50, learning_rate=0.2):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    all_losses = []

    for epoch in range(1, n_epoch + 1):
        total_loss = 0
        indices = list(range(len(training_data)))
        random.shuffle(indices)

        for i in range(0, len(indices), n_batch_size):
            batch_indices = indices[i:i + n_batch_size]
            batch_loss = 0

            optimizer.zero_grad()

            for idx in batch_indices:
                label_tensor, text_tensor, _, _ = training_data[idx]

                # 前向传播
                output = rnn(text_tensor)  # 形状应该是 (1, num_classes)

                # 调整维度
                output = output.view(1, -1)  # 确保形状是 (1, num_classes)
                label = label_tensor.view(1)  # 确保形状是 (1)

                # 计算损失
                loss = criterion(output, label)
                batch_loss += loss

            # 平均损失并反向传播
            batch_loss /= len(batch_indices)
            batch_loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)

            optimizer.step()
            total_loss += batch_loss.item()

        avg_loss = total_loss / (len(indices) / n_batch_size)
        all_losses.append(avg_loss)

        if epoch % report_every == 0:
            print(f"Epoch {epoch}: loss = {avg_loss:.4f}")

    return all_losses

start=time.time()
all_losses=train(rnn,train_set,n_epoch=27,learning_rate=0.15,report_every=5)
end=time.time()
print(f'training took {end-start}')


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
plt.show()



# 结果评估
"""
评估函数：传入模型，测试数据，类别
创建一个confusion报错评估结果
打开评估模式
关闭自动更新梯度
for循环取出数据
将测试数据丢给模型
计算损失
记录损失
归一化

"""
def evaluate(rnn,testing_data,classes):
    confusion=torch.zeros(len(classes),len(classes))
    rnn.eval()
    with torch.no_grad():
        for i in range(len(testing_data)):
            (label_tensor,text_tensor,label,text)=testing_data[i]
            output=rnn(label_tensor)
            guess,guess_i=label_from_attrs(output,classes)
            label_i=classes.index(label)
            confusion[label_i][guess_i]+1

    for i in range(len(classes)):
        denom=confusion[i].sum()
        if denom>0:
            confusion[i]=confusion[i]/denom

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.cpu().numpy())  # numpy uses cpu here so we need to use a cpu version
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

evaluate(rnn, test_set, classes=alldata.labels_uniq)




