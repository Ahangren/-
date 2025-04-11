# 位置编码模块
import math

import torch
import numpy as np
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout_prob,max_len=5000):
        super().__init__()
        self.dropout=nn.Dropout(dropout_prob)
        # 将位置编码注册到非参数列表，保证它不参与梯度计算
        self.register_buffer("positiona_encodings",get_positional_encoding(d_model,max_len),False)

    def forward(self,x):
        pe=self.positional_encoding[:x.shape[0].detach().requires_grad_(False)]
        x=x+pe
        x=self.dropout(x)

        return x

def get_positional_encoding(d_model,max_len=5000):
    # 生成保存位置编码的tensor
    encodings=torch.zeros(max_len,d_model)
    # 位置索引
    position=torch.arange(0,max_len,dtype=torch.float32).unsqueeze(1)
    # 公式中的i
    two_i=torch.arange(0,d_model,2,dtype=torch.float32)
    # 公式中分母部分
    div_tem=torch.exp(two_i*(-(math.log(10000.0/d_model))))
    # sin位置信息
    encodings[:,0::2]=torch.sin(position*div_tem)
    # cos位置信息
    encodings[:,1::2]=torch.cos(position*div_tem)

    encodings=encodings.unsqueeze(1).requires_grad_(False)
    return encodings

# 测试代码
def _test_positional_encoding():
    import matplotlib.pyplot as plt


    plt.figure(figsize=(15, 5))

    pe = get_positional_encoding(20, 100)

    plt.plot(np.arange(100), pe[:, 0, 4:8].numpy())

    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])

    plt.title("Positional encoding")

    plt.show()

if __name__ == '__main__':
    _test_positional_encoding()