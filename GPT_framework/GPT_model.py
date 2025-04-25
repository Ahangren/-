import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    通过正弦和余弦函数为输入嵌入添加位置信息
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)  # 随机失活层

        # 计算位置编码
        position = torch.arange(max_len).unsqueeze(1)  # 位置序列 [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # 除数项
        pe = torch.zeros(max_len, 1, d_model)  # 初始化位置编码矩阵
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数
        self.register_buffer('pe', pe)  # 注册为缓冲区，不参与梯度更新

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x: 输入张量，形状 [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]  # 添加位置编码
        return self.dropout(x)  # 应用dropout


class TransformerBlock(nn.Module):
    """
    单个Transformer块
    包含掩码多头注意力和前馈网络
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 层归一化
        self.ln1 = nn.LayerNorm(d_model)  # 第一个层归一化（注意力前）
        self.ln2 = nn.LayerNorm(d_model)  # 第二个层归一化（前馈网络前）

        # 多头注意力机制
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        # 前馈网络（包含GELU激活）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),  # 扩展维度
            nn.GELU(),  # GELU激活函数
            nn.Linear(d_ff, d_model),  # 降回原维度
            nn.Dropout(dropout)  # 随机失活
        )
        self.dropout = nn.Dropout(dropout)  # 残差连接的dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 自注意力部分（带残差连接和层归一化）
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)  # 自注意力计算
        x = x + self.dropout(attn_output)  # 残差连接
        x = self.ln1(x)  # 层归一化

        # 前馈网络部分（带残差连接和层归一化）
        ffn_output = self.ffn(x)  # 前馈网络计算
        x = x + self.dropout(ffn_output)  # 残差连接
        x = self.ln2(x)  # 层归一化

        return x


class Encoder(nn.Module):
    """
    Transformer编码器
    由多个Transformer块堆叠而成
    """

    def __init__(self, n_layers: int, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 创建多个Transformer块
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 逐层处理输入
        for layer in self.layers:
            x = layer(x, mask)
        return x


class GPT(nn.Module):
    """
    GPT模型（解码器-only架构）
    包含：
    1. 词嵌入 + 位置编码
    2. Transformer编码器堆叠
    3. 输出线性层
    """

    def __init__(self, n_tokens: int, d_model: int, n_layers: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 词嵌入层
        self.token_embed = nn.Embedding(n_tokens, d_model)
        # 位置编码层
        self.pos_embed = PositionalEncoding(d_model, dropout)
        # Transformer编码器
        self.encoder = Encoder(n_layers, d_model, n_head, d_ff, dropout)
        # 输出线性层（词表大小）
        self.generator = nn.Linear(d_model, n_tokens)

        # 初始化权重
        self.apply(self._init_weights)

        # 掩码将在前向传播时创建
        self.mask = None

    def _init_weights(self, module):
        """权重初始化（GPT风格）"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 线性层和嵌入层权重初始化为N(0, 0.02)
            module.weight.data.normal_(mean=0.0, std=0.02)
            # 如果有偏置项，初始化为0
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 创建后续掩码（如果不存在或尺寸不匹配）
        if self.mask is None or self.mask.size(0) != x.size(1):
            self.mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)

        # 获取词嵌入并添加位置编码
        x = self.token_embed(x)  # [batch_size, seq_len, d_model]
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]（Transformer要求的输入格式）
        x = self.pos_embed(x)  # 添加位置编码

        # 通过Transformer编码器
        x = self.encoder(x, self.mask)

        # 生成输出概率
        x = x.transpose(0, 1)  # 恢复为[batch_size, seq_len, d_model]
        x = self.generator(x)  # 线性投影到词表空间

        return x

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """生成上三角掩码（防止看到未来信息）"""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class AdamWarmupCosineDecay(torch.optim.AdamW):
    """
    带预热和余弦衰减的AdamW优化器
    GPT训练常用配置：
    - 初始学习率6e-4
    - 权重衰减0.1（仅应用于特定参数）
    - 预热步骤2000
    """

    def __init__(self, params, lr=6e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.01, warmup=0.1, total_steps=10000):
        super().__init__(params, lr=0.0, betas=betas, eps=eps, weight_decay=weight_decay)
        self.warmup = warmup  # 预热步数比例
        self.total_steps = total_steps  # 总训练步数
        self.base_lr = lr  # 基础学习率
        self.current_step = 0  # 当前步数

    def get_lr(self):
        """计算当前学习率"""
        if self.current_step < self.warmup:
            # 线性预热阶段
            return self.base_lr * (self.current_step / self.warmup)
        # 余弦衰减阶段
        progress = (self.current_step - self.warmup) / (self.total_steps - self.warmup)
        return self.base_lr * (0.5 * (1.0 + math.cos(math.pi * progress)))

    def step(self, closure=None):
        """优化器步进"""
        self.current_step += 1
        # 更新所有参数组的学习率
        for group in self.param_groups:
            group['lr'] = self.get_lr()
        super().step(closure)


def create_optimizer(model: nn.Module, weight_decay: float = 0.1,
                     lr: float = 6e-4, warmup: int = 2000, total_steps: int = 100000):
    """
    创建优化器（GPT风格）
    特点：
    - 权重衰减仅应用于线性层的权重
    - 其他参数（如层归一化参数）不应用权重衰减
    """
    # 收集需要应用权重衰减的参数名
    decay = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn  # 完整参数名
            # 如果是线性层的weight参数，加入衰减集合
            if fpn.endswith('weight') and isinstance(m, nn.Linear):
                decay.add(fpn)

    # 参数分组
    param_dict = {pn: p for pn, p in model.named_parameters()}
    no_decay = set(param_dict.keys()) - decay  # 不需要衰减的参数

    # 创建参数组
    opt_groups = [
        {'params': [param_dict[pn] for pn in sorted(list(decay))], 'weight_decay': weight_decay},
        {'params': [param_dict[pn] for pn in sorted(list(no_decay))], 'weight_decay': 0.0}
    ]

    # 返回配置好的优化器
    return AdamWarmupCosineDecay(opt_groups, lr=lr, warmup=warmup, total_steps=total_steps)


# 示例用法
if __name__ == "__main__":
    # 超参数配置（GPT-small规模）
    n_tokens = 10000  # 词表大小
    d_model = 768  # 嵌入维度
    n_layers = 12  # Transformer层数
    n_head = 12  # 注意力头数
    d_ff = 3072  # 前馈网络隐藏层维度
    dropout = 0.1  # dropout率

    # 创建模型实例
    model = GPT(n_tokens, d_model, n_layers, n_head, d_ff, dropout)

    # 创建优化器（GPT训练配置）
    optimizer = create_optimizer(model)

    # 模拟输入数据（batch_size=4, seq_len=32）
    batch_size = 4
    seq_len = 32
    x = torch.randint(0, n_tokens, (batch_size, seq_len))

    # 前向传播
    output = model(x)
    print(f"输出形状: {output.shape}")  # 应为 [batch_size, seq_len, n_tokens]