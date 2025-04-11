from typing import List,Optional

import torch
import torch.nn as nn

from labml_helpers.module import Module
from labml_nn.utils import clone_module_list
from relative_bullish_attention import RelativeMultiHeadAttention
from transformer编码器和解码器模型.feed_forward import FeedForward


# 定义单层的transformerXL，他在标准的transformer的基础上加上了相对位置编码和记忆机制
class TransformerXLLayer(Module):
    def __init__(self,d_model,self_attn,feed_forward,dropout_prob):
        super().__init__()
        self.size=d_model
        # 带相对位置编码的自注意力模块
        self.self_attn=self_attn
        # 前馈神经网络层FNN
        self.feed_forward=feed_forward
        self.dropout=nn.Dropout(dropout_prob)
        # 自注意力之前的归一化层
        self.norm_self_attn=nn.LayerNorm([d_model])
        # 进入前馈网络前的归一化层
        self.norm_ff=nn.LayerNorm([d_model])

    def forward(self,*,x,mem,mask):
        # 分别对当前输入x和记忆mem（上一层保存下来的输入x）分别进行层归一化
        z=self.norm_self_attn(x)
        if mem is not None:
            mem=self.norm_self_attn(mem)
            # 将记忆mem和当前x沿sew_len维度拼接
            m_z=torch.cat((mem,z),dim=0)
        else:
            m_z=z
        # 进行带记忆和相对位置编码的自注意力计算，只有q是本次的，其他的包含历史信息，mask：掩码矩阵
        self_attn=self.self_attn(query=z,key=m_z,value=m_z,mask=mask)
        # 残差连接：将注意力输出与原始x相加，并通过dropout层
        x=x+self.dropout(self_attn)

        x=self.norm_ff(x)
        ff=self.feed_forward(x)
        x=x+self.dropout(ff)

        return x

class TransformerXL(Module):
    def __init__(self,layer,n_layers):
        super().__init__()
        # 深拷贝layer模块n_layers次，确保各层参数独立
        self.layers=clone_module_list(layer,n_layers)
        # 对最后一层输出进行归一化，提示训练稳定性
        self.norm=nn.LayerNorm([layer.size])

    def forward(self,x,mem,mask):
        # 保存记忆列表，用于缓存每一层当前时间步的输出，作为下一时间步的记忆。
        new_mem=[]

        for i,layer in enumerate(self.layers):
            # 每层处理前，将当前输入x的副本（.detach()避免梯度回传）存入new_mem
            new_mem.append(x.detach())
            #  如果记忆存在，则传入对应层的记忆
            m=mem[i] if mem else None
            x=layer(x=x,mem=m,mask=mask)

            return self.norm(x),new_mem
    
    """
    3. Transformer-XL 的工作机制
    (1) 记忆流动
    记忆结构：mem 和 new_mem 是每层独立的记忆列表，形状为 [mem_len, batch_size, d_model]。
    例如，n_layers=6 时，mem 是长度为6的列表，每元素对应一层的历史记忆。
    更新逻辑：
    每个时间步，当前输入经过各层处理，同时每层的输入被缓存为下一时间步的记忆。
    记忆通过 .detach() 截断梯度，防止跨时间步反向传播。
    (2) 相对位置编码的优势
    由于每层的输入可能包含历史记忆（mem 与当前 x 拼接），绝对位置编码会失效。
    相对位置编码（如 RelativeMultiHeadAttention）通过建模 i-j 的相对偏移，天然支持变长序列。
    (3) 计算效率
    记忆复用：避免像标准 Transformer 那样对历史序列重复计算，显著降低长序列的计算成本。
    并行性：尽管记忆机制依赖时间步，但单个时间步内的层间计算完全并行。

    """