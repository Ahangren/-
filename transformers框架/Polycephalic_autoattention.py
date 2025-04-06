# 一、导入相关模块
import math
from typing import Optional,List
import torch
from torch import nn
from labml import tracker

# 二、为多头关注做好准备
# 此模块执行线性变换，并将向量拆分为给定数量的头，以实现多头关注。这用于转换键、查询和值向量。
class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self,d_model,heads,d_k,bias):
        super().__init__()
        # 线性变换层
        self.linear=nn.Linear(d_model,heads*d_k,bias=bias)
        # 头数量
        self.heads=heads
        # 每个头中的向量维度
        self.d_k=d_k

    def forward(self,x,):
        # 一般传过来的向量维度为[seq_len, batch_size,d_model]或[batch_size,d_model]
        head_shape=x.shape[:-1]
        # 因为传来的最后一个维度都是d_model，所以我们要对最后一个维度使用线性变换层，并将其拆分为heads
        x=self.linear(x)
        # 将最后一个维度拆分为多个head,x.view(*head_shape, heads, d_k) → 新形状为 (batch_size, seq_len, heads, d_k)
        x=x.view(*head_shape,self.heads,self.d_k)
        # 输出具有 shape 或[seq_len, batch_size, heads, d_k][batch_size, heads, d_model]
        return x


