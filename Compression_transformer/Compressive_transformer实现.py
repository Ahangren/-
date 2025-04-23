# 导入模块
from typing import Optional,List
import torch
import torch.nn as nn
import torch.nn.functional as F

from labml_helpers.module import Module,TypedModuleList
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
from labml_nn.transformers.xl.relative_mha import RelativeMultiHeadAttention
from labml_nn.utils import clone_module_list

from transorflow_preject.leNet_5 import batch_size


class Conv1dCompression(Module):
    """
    1D卷积压缩层
    这是对PyTorch中1D卷积的封装，用于实现记忆压缩功能
    功能：通过卷积操作将长序列的记忆压缩为更短的表示，减少计算量同时保留关键信息
    """

    def __init__(self,compression_rate:int,d_model:int):
        """

        :param compression_rate: 压缩率，决定压缩的程度
        :param d_model: 特征维度 - 特征向量的维度
        """
        super().__init__()
        # 使用1D卷积进行压缩，卷积核的大小和步长都等于压缩率
        self.conv1d=nn.Conv1d(d_model,d_model,kernel_size=compression_rate,stride=compression_rate)

    def forward(self,mem:torch.Tensor):
        """
        前向传播
        :param mem: 相当于掩码，形状为：（seq_len，batch_size，d_model）
        :return: 处理过的记忆
        """
        # 调整维度顺序以适应卷积的输入要求（batch_size，features，d_model）
        mem=mem.permute(1,2,0)
        # 通过卷积进行压缩
        c_mem=self.conv1d(mem)
        # 恢复原始顺序
        return c_mem.permute(2,0,1)


class CompressiveTransformerLayer(Module):
    """
    压缩Transformer层
    实现了单个压缩Transformer层的功能，结合了自注意力机制和前馈网络
    功能：这是Transformer的核心计算单元，通过引入了记忆压缩机制，可以处理更长的序列依赖
    """

    def __init__(self,d_model:int,self_attn:RelativeMultiHeadAttention,feed_forward:FeedForward,dropout:float,compress:Conv1dCompression):
        """
        初始化参数
        :param d_model: 词嵌入维度
        :param self_attn:  相对位置多头注意力机制
        :param feed_forward:  前馈神经网络
        :param dropout: dropout概率
        :param compress: 压缩函数
        """
        super().__init__()
        self.d_model=d_model
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.dropout=nn.Dropout(dropout)
        self.compress=compress

        self.norm_self_attn=nn.LayerNorm([d_model])
        self.norm_ff=nn.LayerNorm([d_model])

    def concat_memory(self,z:torch.Tensor,mem:Optional[torch.Tensor],c_mem:Optional[torch.Tensor]):
        """
        拼接归一化后的词嵌入与记忆（压缩和未压缩的），这是实现长距离依赖的关键
        通过将当前输入与历史记忆结合
        :param z: 当前序列
        :param mem: 主记忆
        :param c_mem: 历史记忆
        :return:
        """
        if mem is None:
            return z

        if c_mem is not None:
            mem=torch.cat((mem,c_mem),dim=0)

        # 对记忆进行归一化
        mem=self.norm_self_attn(mem)
        # 返回拼接后的记忆和当前输入
        return torch.cat((mem,z),dim=0)

    def forward(self,*,x:torch.Tensor,mem:Optional[torch.Tensor],c_mem:Optional[torch.Tensor],mask:torch.Tensor):
        """
        前向传播
        :param x: 输入当前序列（seq_len，batch_size，d_model）
        :param mem: 历史记忆（mem_len，batch_size，d_model）
        :param c_mem: 压缩记忆（c_mem_len，batch_size，d_model）
        :param mask: 注意力掩码
        :return:
        """
        # 对输入层进行归一化
        z=self.norm_self_attn(x)
        # 拼接记忆和压缩记忆
        m_z=self.concat_memory(z,mem,c_mem)
        # 计算注意力
        self_attn=self.self_attn(query=z,key=m_z,value=m_z,mask=mask)
        # 残差连接和dropout
        x=x+self.dropout(self_attn)
        # 前馈网络
        z = self.norm_ff(x)
        z=self.feed_forward(z)

        # 残差连接和dropout
        x=x+self.dropout(z)
        return x

class CompressiveTransformer(Module):
    """
    压缩Transformer模型，由多个Transformer层堆叠而成
    这是完整的模型架构，通过堆叠多个处理层来构建深度网络，每层都有自己的记忆压缩机制
    """
    def __init__(self,layer:CompressiveTransformerLayer,n_layers:int):
        super().__init__()
        # 克隆多个相同层
        self.layers=clone_module_list(layer,n_layers)
        # 最终归一化
        self.norm=nn.LayerNorm([layer.size])

    def forward(self,x:torch.Tensor,mem:List[torch.Tensor],c_mem:List[torch.TEnsor],mask:torch.Tensor):
        """
        前向传播
        :param x: 输入词嵌入（seq_len，batch_size，d_model）
        :param mem: 历史记忆（mem_len，batch_size,d_model）
        :param c_mem: 压缩记忆（c_mem_len, batch_size ,d_model)
        :param mask: 掩码
        :return: 训练好的词嵌入矩阵
        """
        # 储存新的记忆用于下一批次
        new_mem=[]
        # 逐层处理
        for i, layer in enumerate(self.layers):
            # 保存当前步的输出作为下一时间步的记忆
            new_mem.append(x.detach())
            # 获取当前层的记忆
            mem=mem[i] if mem else None
            c_mem=c_mem[i] if c_mem else None
            # 通过当前层处理
            x=layer(x,mem,c_mem,mask)
            # 最终归一化
        return self.norm(x),new_mem

class AttentionReconstructionLoss:
    """
    注意力重建损失
    通过比较压缩记忆和原始记忆的注意力输出差异来计算损失
    专门用于训练压缩函数
    这是一种辅助目标，确保压缩后的记忆仍能保持原始记忆的关键信息
    特别是对于注意力计算相关的信息
    """
    def __init__(self,layers:TypedModuleList[CompressiveTransformerLayer]):
        """
        初始化
        :param layers: 压缩的Transformer层列表
        """
        self.layers=layers
        self.loss_func=nn.MSELoss

    def prepare_for_attn(self,pmha:PrepareForMultiHeadAttention,x:torch.Tensor):
        """
        准备注意力计算的输入，冻结压缩函数外的参数

        """
        # 获取输入的形状
        head_shape=x.shape[:-1]
        # 冻结线性变换参数
        weight=pmha.linear.weight.detach()
        bias=pmha.linear.bias.detach() if pmha.linear.bias is not None else None
        x=F.linear(x,weight=weight,bias=bias)
        # 分割多头
        x=x.view(*head_shape,pmha.heads,pmha.d_k)
        return x

    def attn(self,layer:RelativeMultiHeadAttention,query:torch.Tensor,key:torch.Tensor,value:torch.Tensor):
        """计算注意力，冻结除压缩函数外的参数"""
        # 准备qkv
        query=self.prepare_for_attn(layer.query,query)
        key=self.prepare_for_attn(layer.key,key)
        value=self.prepare_for_attn(layer.value,value)

        # 计算注意力分数
        scores=torch.einsum('ibhd,jbhd->ijbh',query,key)
        scores*=layer.scale

        # softmax归一化
        attn=layer.softmax(scores)

        return torch.enisum('ijbh,jbhd->ibhd',attn,value)

    def norm(self,ln:nn.LayerNorm,x:torch.Tensor):
        """层归一化，冻结参数"""
        weight=ln.weight.detach() if ln.weight is not None else None
        bias = ln.bias.detach() if ln.bias is not None else None
        return F.layer_norm(x,ln.normalized_shape,weight,bias,ln.eps)

    def calc_loss(self,layer:CompressiveTransformerLayer,h:torch.Tensor,mem:torch.Tensor):
        """单层损失"""
        # 冻结输入
        h=h.detach()
        mem=mem.detach()

        # 压缩记忆（这是唯一不冻结的操作）
        c_mem=layer.compress(mem)

        # 归一化处理
        h=self.norm(layer.norm_self_attn,h)
        mem=self.norm(layer.norm_self_attn,mem)
        c_mem=self.norm(layer.norm_self_attn,c_mem)

        # 计算两种记忆的注意力输出
        attn_mem=self.attn(layer.self_attn,h,mem,mem)
        attn_cmem=self.attn(layer.self_attn,h,c_mem,c_mem)

        # 计算均方误差
        return self.loss_func(attn_cmem,attn_mem)

    def __call__(self, h:List[torch.Tensor],mem:List[torch.Tensor]):
        losses=[self.calc_loss(layer,h[n],mem[n]) for n,layer in enumerate(self.layers)]
        return sum(losses)




















