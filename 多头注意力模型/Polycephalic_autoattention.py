# 一、导入相关模块
import math
from typing import Optional, List
import torch
from torch import nn
from labml import tracker


# 二、为多头关注做好准备
# 此模块执行线性变换，并将向量拆分为给定数量的头，以实现多头关注。这用于转换键、查询和值向量。
class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, d_k, bias):
        super().__init__()
        # 线性变换层
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # 头数量
        self.heads = heads
        # 每个头中的向量维度
        self.d_k = d_k

    def forward(self, x, ):
        # 一般传过来的向量维度为[seq_len, batch_size,d_model]或[batch_size,d_model]
        head_shape = x.shape[:-1]
        # 因为传来的最后一个维度都是d_model，所以我们要对最后一个维度使用线性变换层，并将其拆分为heads
        x = self.linear(x)
        # 将最后一个维度拆分为多个head,x.view(*head_shape, heads, d_k) → 新形状为 (batch_size, seq_len, heads, d_k)
        x = x.view(*head_shape, self.heads, self.d_k)
        # 输出具有 shape 或[seq_len, batch_size, heads, d_k][batch_size, heads, d_model]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout_prob=0.1, bias=True):
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        # 生成qkv并且转换成多头形式
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
        # softmax对行维度进行处理
        self.softmax = nn.Softmax(dim=1)
        # 定义输出层，这里要注意，要保证传入的数据的shape什么样这里就要返回什么样
        self.output = nn.Linear(d_model, d_model)
        # dropout层
        self.dropout = nn.Dropout(dropout_prob)
        # softmax之前的缩放因子
        self.scale = 1 / math.sqrt(self.d_k)
        # 我们存储 attentions，以便在需要时将其用于日志记录或其他计算
        self.attn = None

    def get_score(self, query, key):

        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    # 生成掩码函数
    def prepare_mask(self, mask, query_shape, key_shape):
        # mask具有shape ，其中第一个维度是查询维度。如果查询维度等于[seq_len_q, seq_len_k, batch_size]1它将会被广播。
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        # 应用所有头部的相同模板，生成模板形状：[seq_len_q,seq_len_k,batch_size,heads]
        mask = mask.unsqueeze(-1)

        return mask

    def forward(self, query, key, value, mask):
        seq_len, batch_size, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_score(query, key)

        scores *= self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.softmax(scores)
        tracker.debug('attn', attn)
        attn = self.dropout(attn)
        x = torch.einsum('ijbh,jbhd->ibhd', attn, value)
        self.attn = attn.detach()
        x = x.reshape(seq_len, batch_size, -1)

        return self.output(x)
