import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """位置编码层，为输入序列添加位置信息"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)  # 形状改为 [max_len, 1, d_model] 以便广播
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入x形状应为 [seq_len, batch_size, d_model]"""
        return x + self.pe[:x.size(0)]


class TokenEmbedding(nn.Module):
    """词嵌入层，将token转换为向量表示"""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """输入tokens形状应为 [seq_len, batch_size]"""
        return self.embedding(tokens.long()) * math.sqrt(self.d_model)


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器单层结构"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.is_save_ff_input = False
        self.ff_input = None

    def forward(self, src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        # 自注意力部分
        attn_output, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # 前馈网络部分
        if self.is_save_ff_input:
            self.ff_input = src.detach().clone()

        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    """Transformer编码器，由多个编码层堆叠而成"""

    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        for layer in self.layers:
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return src


class Generator(nn.Module):
    """生成器层，将编码器输出转换为词表概率分布"""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入x形状应为 [seq_len, batch_size, d_model]"""
        return self.proj(x)


class AutoregressiveModel(nn.Module):
    """自回归Transformer模型"""

    def __init__(self, src_embed: nn.Module, encoder: nn.Module, generator: nn.Module,
                 is_save_ff_input: bool = False):
        super().__init__()
        self.src_embed = src_embed
        self.pos_encoder = PositionalEncoding(src_embed.d_model)
        self.encoder = encoder
        self.generator = generator

        if len(self.encoder.layers) > 0:
            self.encoder.layers[-1].is_save_ff_input = is_save_ff_input

        self.register_buffer('src_mask', None)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成自注意力掩码 [sz, sz]"""
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

    def forward(self, src: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None):
        """
        前向传播
        参数:
            src: 输入序列 [seq_len, batch_size]
            src_key_padding_mask: 可选，形状应为 [batch_size, seq_len]
        """
        # 确保输入是 [seq_len, batch_size] 形状
        if src.dim() == 2 and src.size(0) != src.size(1):
            pass  # 已经是正确形状
        else:
            raise ValueError(f"输入src的形状应为[seq_len, batch_size]，但得到的是{src.shape}")

        seq_len, batch_size = src.size(0), src.size(1)

        # 生成注意力掩码
        if self.src_mask is None or self.src_mask.size(0) != seq_len:
            self.src_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)

        # 词嵌入 + 位置编码
        src_emb = self.pos_encoder(self.src_embed(src))

        # 检查key_padding_mask形状
        if src_key_padding_mask is not None:
            if src_key_padding_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"key_padding_mask的形状应为[batch_size, seq_len]={[batch_size, seq_len]}，"
                    f"但得到的是{src_key_padding_mask.shape}"
                )

        # 通过编码器
        memory = self.encoder(
            src_emb,
            mask=self.src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # 生成预测
        output = self.generator(memory)

        return output, None


# 配置类
class ModelConfig:
    def __init__(self):
        # 模型参数
        self.vocab_size = 10000
        self.d_model = 256
        self.dim_feedforward = 1024
        self.nhead = 8
        self.num_layers = 6
        self.dropout = 0.1

        # 训练参数
        self.batch_size = 6
        self.learning_rate = 1.0
        self.seq_len = 1024


# 使用示例
if __name__ == "__main__":
    config = ModelConfig()

    # 构建模型组件
    src_embed = TokenEmbedding(config.vocab_size, config.d_model)
    encoder_layer = TransformerEncoderLayer(config.d_model, config.nhead,
                                            config.dim_feedforward, config.dropout)
    encoder = TransformerEncoder(encoder_layer, config.num_layers)
    generator = Generator(config.d_model, config.vocab_size)
    model = AutoregressiveModel(src_embed, encoder, generator)

    # 创建输入 [seq_len, batch_size]
    input_seq = torch.randint(0, config.vocab_size, (config.seq_len, config.batch_size))

    # 可选：创建key_padding_mask [batch_size, seq_len]
    key_padding_mask = torch.zeros(config.batch_size, config.seq_len).bool()

    # 前向传播
    output, _ = model(input_seq, src_key_padding_mask=key_padding_mask)
    print(f"输出形状: {output.shape}")  # 应该是 [seq_len, batch_size, vocab_size]