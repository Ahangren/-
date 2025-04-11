import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout,activation=nn.ReLU(),is_gated=False,bias1=True,bias2=True,bias_grate=True):
        super().__init__()
        # 第一层
        self.layer1=nn.Linear(d_model,d_ff,bias=bias1)
        # 第二层
        self.layer2=nn.Linear(d_ff,d_model,bias=bias2)
        # dropout层
        self.dropout=nn.Dropout(dropout)
        # 激活函数
        self.activation=activation
        # 是否有门（比如激活函数是GeLU的话就有）
        self.is_gated=is_gated

        if is_gated:
            self.linear_v=nn.Linear(d_model,d_ff,bias=bias_grate)

    def forward(self,x):

        g=self.activation(self.layer1(x))
        if self.is_gated:
            x=g*self.linear_v(x)
        else:
            x=g

        x=self.dropout(x)
        return self.layer2(x)

