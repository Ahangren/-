import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple


class RelativeMultiHeadAttention(nn.Module):
    """实现相对位置编码的多头注意力机制"""

    def __init__(self, d_model: int, heads: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        # 确保特征维度能被头数整除
        assert d_model % heads == 0, "d_model must be divisible by heads"

        # 初始化参数
        self.d_k = d_model // heads  # 每个头的维度
        self.heads = heads  # 注意力头数
        self.scale = 1.0 / math.sqrt(self.d_k)  # 缩放因子
        self.max_len = max_len  # 最大序列长度

        # 定义线性变换层
        self.to_q = nn.Linear(d_model, d_model)  # 查询变换
        self.to_k = nn.Linear(d_model, d_model)  # 键变换
        self.to_v = nn.Linear(d_model, d_model)  # 值变换
        self.to_out = nn.Linear(d_model, d_model)  # 输出变换

        # 相对位置编码表 (2*max_len-1, heads)
        # 可学习参数，表示不同相对位置的关系
        self.relative_pos_table = nn.Parameter(torch.randn(2 * max_len - 1, heads))

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # 获取输入形状参数
        batch_size = query.size(1)  # 批大小
        q_len = query.size(0)  # 查询序列长度
        k_len = key.size(0)  # 键序列长度

        # 线性变换 + 分头处理
        # 将输入通过线性层并重塑为多头形式
        q = self.to_q(query).view(q_len, batch_size, self.heads, self.d_k)
        k = self.to_k(key).view(k_len, batch_size, self.heads, self.d_k)
        v = self.to_v(value).view(k_len, batch_size, self.heads, self.d_k)

        # 计算注意力分数 [batch, heads, q_len, k_len]
        # 使用爱因斯坦求和约定高效计算
        scores = torch.einsum('qbhd,kbhd->bhqk', q, k) * self.scale

        # 生成相对位置索引 [q_len, k_len]
        range_q = torch.arange(q_len, device=query.device)[:, None]  # 查询位置索引
        range_k = torch.arange(k_len, device=key.device)[None, :]  # 键位置索引
        distance = (range_q - range_k) + (self.max_len - 1)  # 相对位置距离，确保非负

        # 从表中获取相对位置偏置 [q_len, k_len, heads]
        # 使用clamp确保索引不越界
        rel_pos_bias = self.relative_pos_table[distance.clamp(0, 2 * self.max_len - 2)]

        # 将位置偏置加到注意力分数上
        # 调整维度顺序以匹配scores的形状
        scores = scores + rel_pos_bias.permute(2, 0, 1).unsqueeze(0)

        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)  # 应用dropout

        # 加权求和 [q_len, batch, heads, d_k]
        out = torch.einsum('bhqk,kbhd->qbhd', attn, v)

        # 合并多头 [q_len, batch, d_model]
        out = out.reshape(q_len, batch_size, -1)
        return self.to_out(out)  # 通过输出线性层


class FeedForward(nn.Module):
    """Transformer的前馈神经网络"""

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        # 定义两层线性变换，中间有GELU激活和Dropout
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),  # 扩展维度
            nn.GELU(),  # GELU激活函数
            nn.Dropout(dropout),  # Dropout
            nn.Linear(d_ff, d_model),  # 恢复维度
            nn.Dropout(dropout)  # Dropout
        )

    def forward(self, x):
        return self.net(x)  # 顺序执行定义的前馈网络


class Conv1dCompression(nn.Module):
    """使用1D卷积进行记忆压缩"""

    def __init__(self, compression_rate: int, d_model: int):
        super().__init__()
        # 定义1D卷积层
        self.conv = nn.Conv1d(
            in_channels=d_model,  # 输入通道数
            out_channels=d_model,  # 输出通道数
            kernel_size=compression_rate,  # 卷积核大小
            stride=compression_rate  # 步长
        )

    def forward(self, mem):
        # 输入形状: [seq_len, batch, dim]
        # 调整维度顺序以适应卷积层
        mem = mem.permute(1, 2, 0)  # [batch, dim, seq_len]
        # 应用卷积压缩
        compressed = self.conv(mem)  # [batch, dim, compressed_len]
        # 恢复原始维度顺序
        return compressed.permute(2, 0, 1)  # [compressed_len, batch, dim]


class CompressiveTransformerLayer(nn.Module):
    """压缩Transformer的单个层"""

    def __init__(self, d_model: int, heads: int, compression_rate: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        # 初始化各组件
        self.attn = RelativeMultiHeadAttention(d_model, heads, max_len, dropout)  # 注意力
        self.ff = FeedForward(d_model, dropout=dropout)  # 前馈网络
        self.compress = Conv1dCompression(compression_rate, d_model)  # 压缩模块

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)  # 自注意力前归一化
        self.norm2 = nn.LayerNorm(d_model)  # 前馈前归一化

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mem=None, c_mem=None, mask=None):
        # 自注意力部分
        # 计算带记忆的注意力输出
        attn_out = self._attention_block(x, mem, c_mem, mask)
        # 残差连接和dropout
        x = x + self.dropout(attn_out)

        # 前馈部分
        # 通过前馈网络
        ff_out = self.ff(self.norm2(x))
        # 残差连接和dropout
        x = x + self.dropout(ff_out)
        return x

    def _attention_block(self, x, mem, c_mem, mask):
        """处理记忆拼接的注意力计算"""
        # 归一化当前输入
        norm_x = self.norm1(x)

        # 如果有记忆，处理记忆
        if mem is not None:
            mem = self.norm1(mem)  # 归一化记忆
            if c_mem is not None:
                c_mem = self.norm1(c_mem)  # 归一化压缩记忆
                mem = torch.cat([mem, c_mem], dim=0)  # 拼接记忆
            # 将记忆与当前输入拼接
            norm_x = torch.cat([mem, norm_x], dim=0)

        # 只对当前输入部分计算query
        return self.attn(norm_x[-x.size(0):], norm_x, norm_x, mask)


class CompressiveTransformer(nn.Module):
    """完整的压缩Transformer模型"""

    def __init__(self, n_layers: int, d_model: int, heads: int, compression_rate: int, max_len: int = 512):
        super().__init__()
        # 创建多层Transformer
        self.layers = nn.ModuleList([
            CompressiveTransformerLayer(d_model, heads, compression_rate, max_len)
            for _ in range(n_layers)
        ])
        # 最终归一化层
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mem_list=None, c_mem_list=None, mask=None):
        # 存储新的记忆
        new_mems = []
        # 逐层处理
        for i, layer in enumerate(self.layers):
            # 获取当前层的记忆
            mem = mem_list[i] if mem_list is not None else None
            c_mem = c_mem_list[i] if c_mem_list is not None else None

            # 存储当前状态作为新记忆
            new_mems.append(x.detach())

            # 通过当前层处理
            x = layer(x, mem, c_mem, mask)

        # 最终归一化
        return self.norm(x), new_mems


# 测试代码
if __name__ == "__main__":
    # 模型参数配置
    n_layers = 6  # Transformer层数
    d_model = 512  # 特征维度
    heads = 8  # 注意力头数
    compression_rate = 2  # 压缩率
    seq_len = 10  # 输入序列长度
    batch_size = 32  # 批大小
    max_len = 512  # 最大序列长度

    # 初始化模型
    model = CompressiveTransformer(n_layers, d_model, heads, compression_rate, max_len)

    # 模拟输入数据
    x = torch.randn(seq_len, batch_size, d_model)  # 输入序列 [seq_len, batch, dim]
    mem_list = [torch.randn(5, batch_size, d_model) for _ in range(n_layers)]  # 记忆列表
    c_mem_list = [torch.randn(3, batch_size, d_model) for _ in range(n_layers)]  # 压缩记忆列表

    # 前向传播测试
    output, new_mems = model(x, mem_list, c_mem_list)
    print(f"Output shape: {output.shape}")  # 预期输出 [10, 32, 512]
    print("模型执行成功！")