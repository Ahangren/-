import math
from typing import Set, Optional, Tuple
import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    """优化的旋转位置编码(RoPE)实现"""

    def __init__(self, d: int, base: int = 10000):
        super().__init__()
        # 正确初始化theta参数: θ_i = 10000^(-2i/d)
        theta = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer("theta", theta, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_heads, d = x.shape
        d_2 = d // 2

        # 生成位置索引[0, 1, ..., seq_len-1]
        pos_idx = torch.arange(seq_len, device=x.device, dtype=torch.float32)

        # 计算旋转角度: pos_idx * theta
        freqs = torch.einsum('i,j->ij', pos_idx, self.theta)

        # 拼接相同的旋转角度以匹配x的维度
        freqs = torch.cat([freqs, freqs], dim=-1)  # [seq_len, d]

        # 一次性计算所有cos和sin值
        cos_val = torch.cos(freqs)[None, :, None, :]  # [1, seq_len, 1, d]
        sin_val = torch.sin(freqs)[None, :, None, :]  # [1, seq_len, 1, d]

        # 旋转操作: [-x_{d/2+1}, ..., -x_d, x_1, ..., x_{d/2}]
        x_rot = torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

        # 应用旋转位置编码
        return x * cos_val + x_rot * sin_val


class SelfAttention(nn.Module):
    """优化的自注意力层实现"""

    def __init__(self, d_model: int, n_heads: int, d_k: int, is_causal: bool, eps: float = 1e-5):
        super().__init__()
        self.is_causal = is_causal
        self.n_heads = n_heads
        self.d_k = d_k
        self.scale = 1 / math.sqrt(d_k)

        # 线性投影层
        self.q_proj = nn.Linear(d_model, n_heads * d_k)
        self.k_proj = nn.Linear(d_model, n_heads * d_k)
        self.v_proj = nn.Linear(d_model, n_heads * d_k)

        # 输出层
        self.out_proj = nn.Linear(n_heads * d_k, d_model)

        # 归一化层
        self.norm = nn.LayerNorm(d_model, eps=eps)

        # 旋转位置编码
        self.rotary_pe = RotaryPositionalEmbeddings(d_k)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 使用Xavier初始化查询、键、值投影层
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # 偏置初始化为0
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _create_causal_mask(self, attn: torch.Tensor) -> torch.Tensor:
        """创建因果注意力掩码"""
        if not self.is_causal:
            return attn

        # 生成上三角布尔掩码(对角线以上为True)
        mask = torch.ones_like(attn, dtype=torch.bool).triu(diagonal=1)
        return attn.masked_fill(mask, float('-inf'))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # 残差连接
        residual = h
        h = self.norm(h)

        batch_size, seq_len, _ = h.shape

        # 投影查询、键、值并分割头
        q = self.q_proj(h).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.k_proj(h).view(batch_size, seq_len, self.n_heads, self.d_k)
        v = self.v_proj(h).view(batch_size, seq_len, self.n_heads, self.d_k)

        # 应用旋转位置编码
        q = self.rotary_pe(q)
        k = self.rotary_pe(k)

        # 计算注意力分数
        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale

        # 应用因果掩码(如果需要)
        attn = self._create_causal_mask(attn)

        # 计算注意力权重
        attn = torch.softmax(attn, dim=-1)

        # 应用注意力到值上
        h = torch.einsum("bhij,bjhd->bihd", attn, v)

        # 合并多头并投影输出
        h = h.reshape(batch_size, seq_len, -1)
        h = self.out_proj(h)

        return h + residual


class CrossAttention(nn.Module):
    """优化的交叉注意力层实现"""

    def __init__(self, d_model: int, n_heads: int, d_k: int, eps: float = 1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.scale = 1 / math.sqrt(d_k)

        # 线性投影层
        self.q_proj = nn.Linear(d_model, n_heads * d_k)
        self.k_proj = nn.Linear(d_model, n_heads * d_k)
        self.v_proj = nn.Linear(d_model, n_heads * d_k)

        # 输出层
        self.out_proj = nn.Linear(n_heads * d_k, d_model)

        # 归一化层
        self.norm = nn.LayerNorm(d_model, eps=eps)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, e: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # 残差连接
        residual = e
        e = self.norm(e)

        batch_size, chunks, neighbors, neighbor_len, _ = e.shape
        _, chunk_num, chunk_len, _ = h.shape

        # 投影查询(来自编码器输出)
        q = self.q_proj(e).view(batch_size, chunks, neighbors, neighbor_len, self.n_heads, self.d_k)

        # 投影键和值(来自输入块)
        k = self.k_proj(h).view(batch_size, chunk_num, chunk_len, self.n_heads, self.d_k)
        v = self.v_proj(h).view(batch_size, chunk_num, chunk_len, self.n_heads, self.d_k)

        # 计算注意力分数
        attn = torch.einsum('bcnihd,bcjhd->bcnhij', q, k) * self.scale

        # 计算注意力权重
        attn = torch.softmax(attn, dim=-1)

        # 应用注意力到值上
        e = torch.einsum("bcnhij,bcjhd->bcnihd", attn, v)

        # 合并多头并投影输出
        e = e.reshape(batch_size, chunks, neighbors, neighbor_len, -1)
        e = self.out_proj(e)

        return e + residual


class ChunkedCrossAttention(nn.Module):
    """优化的分块交叉注意力实现"""

    def __init__(self, d_model: int, n_heads: int, d_k: int, chunk_len: int, eps: float = 1e-5):
        super().__init__()
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_k = d_k
        self.scale = 1 / math.sqrt(d_k)

        # 线性投影层
        self.q_proj = nn.Linear(d_model, n_heads * d_k)
        self.k_proj = nn.Linear(d_model, n_heads * d_k)
        self.v_proj = nn.Linear(d_model, n_heads * d_k)

        # 输出层
        self.out_proj = nn.Linear(n_heads * d_k, d_model)

        # 归一化层
        self.norm = nn.LayerNorm(d_model, eps=eps)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # 残差连接
        residual = h

        # 处理空块情况
        if e.shape[1] == 0:  # 如果没有块
            return residual

        batch_size = h.size(0)
        chunks = e.size(1)
        neighbors = e.size(2)
        neighbor_len = e.size(3)
        d_model = h.size(-1)

        # 移位序列以对齐块
        h_shifted = h[:, self.chunk_len - 1:]
        h_shifted = self.norm(h_shifted)

        # 计算需要的填充长度
        total_len = chunks * self.chunk_len
        current_len = h_shifted.size(1)
        pad_len = max(0, total_len - current_len)

        # 应用填充(如果需要)
        if pad_len > 0:
            h_padded = torch.cat([
                h_shifted,
                torch.zeros(batch_size, pad_len, d_model, device=h.device)
            ], dim=1)
        else:
            h_padded = h_shifted[:, :total_len]

        # 重塑为块形式
        h_blocks = h_padded.view(batch_size, chunks, self.chunk_len, d_model)

        # 投影查询(来自输入)
        q = self.q_proj(h_blocks).view(batch_size, chunks, self.chunk_len, self.n_heads, self.d_k)

        # 投影键和值(来自编码的邻居)
        k = self.k_proj(e).view(batch_size, chunks, neighbors, neighbor_len, self.n_heads, self.d_k)
        v = self.v_proj(e).view(batch_size, chunks, neighbors, neighbor_len, self.n_heads, self.d_k)

        # 计算注意力分数
        attn = torch.einsum('bcihd,bcnjhd->bchinj', q, k) * self.scale

        # 计算注意力权重
        attn = torch.softmax(attn.view(*attn.shape[:-2], -1), dim=-1).view(attn.shape)

        # 应用注意力到值上
        h_out = torch.einsum("bchinj,bcnjhd->bcihd", attn, v)

        # 合并多头并投影输出
        h_out = h_out.reshape(batch_size, chunks * self.chunk_len, -1)
        h_out = self.out_proj(h_out)

        # 恢复原始序列长度
        h_out = torch.nn.functional.pad(h_out, (0, 0, self.chunk_len - 1, 0))
        h_out = h_out[:, :residual.size(1)]

        return h_out + residual


class FeedForward(nn.Module):
    """优化的前馈网络实现"""

    def __init__(self, d_model: int, d_ff: int, eps: float = 1e-5):
        super().__init__()
        # 两层线性变换
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        # 激活函数(GELU比ReLU表现更好)
        self.activation = nn.GELU()

        # 归一化层
        self.norm = nn.LayerNorm(d_model, eps=eps)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 第一层使用Kaiming初始化配合GELU
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='gelu')
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)

        # 第二层使用Xavier初始化
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # 残差连接
        residual = h
        h = self.norm(h)

        h = self.linear1(h)
        h = self.activation(h)
        h = self.linear2(h)

        return h + residual


class NearestNeighborEncoder(nn.Module):
    """优化的最近邻编码器实现"""

    def __init__(self, chunk_len: int, n_layers: int, ca_layers: Set[int],
                 d_model: int, n_heads: int, d_k: int, d_ff: int, eps: float = 1e-5):
        super().__init__()
        self.chunk_len = chunk_len
        self.ca_layers = ca_layers

        # 自注意力层
        self.self_attn_layers = nn.ModuleList([
            SelfAttention(d_model, n_heads, d_k, is_causal=False, eps=eps)
            for _ in range(n_layers)
        ])

        # 交叉注意力层(只在指定层使用)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(d_model, n_heads, d_k, eps=eps)
            for _ in range(len(ca_layers))
        ])

        # 前馈网络层
        self.ffn_layers = nn.ModuleList([
            FeedForward(d_model, d_ff, eps=eps)
            for _ in range(n_layers)
        ])

        # 输入归一化层
        self.input_norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, e: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        batch_size, chunks, neighbors, neighbor_len, d_model = e.shape

        # 分割输入序列为块
        h_split = h[:, :chunks * self.chunk_len].view(batch_size, chunks, self.chunk_len, d_model)
        h_split = self.input_norm(h_split)

        cross_attn_idx = 0

        for layer_idx in range(len(self.self_attn_layers)):
            # 自注意力
            e = self.self_attn_layers[layer_idx](
                e.view(-1, neighbor_len, d_model)
            ).view(batch_size, chunks, neighbors, neighbor_len, d_model)

            # 交叉注意力(如果当前层需要)
            if layer_idx in self.ca_layers:
                e = self.cross_attn_layers[cross_attn_idx](e, h_split)
                cross_attn_idx += 1

            # 前馈网络
            e = self.ffn_layers[layer_idx](e)

        return e


class RetroModel(nn.Module):
    """优化的RETRO模型实现"""

    def __init__(self, n_vocab: int, d_model: int, n_layers: int, ca_layers: Set[int],
                 chunk_len: int, n_heads: int, d_k: int, d_ff: int,
                 encoder: NearestNeighborEncoder, eps: float = 1e-5):
        super().__init__()
        self.chunk_len = chunk_len
        self.ca_layers = ca_layers
        self.encoder = encoder

        # 词嵌入层
        self.token_emb = nn.Embedding(n_vocab, d_model)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)

        # 自注意力层
        self.self_attn_layers = nn.ModuleList([
            SelfAttention(d_model, n_heads, d_k, is_causal=True, eps=eps)
            for _ in range(n_layers)
        ])

        # 分块交叉注意力层
        self.chunked_cross_attn_layers = nn.ModuleList([
            ChunkedCrossAttention(d_model, n_heads, d_k, chunk_len, eps=eps)
            for _ in range(len(ca_layers))
        ])

        # 前馈网络层
        self.ffn_layers = nn.ModuleList([
            FeedForward(d_model, d_ff, eps=eps)
            for _ in range(n_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, n_vocab)
        nn.init.zeros_(self.output_layer.bias)

        # 编码器输出归一化层
        self.encoder_norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor, ret: torch.Tensor) -> torch.Tensor:
        # 输入验证
        if x.dim() != 2:
            raise ValueError(f"输入x必须是2D张量[batch_size, seq_len], 但得到{x.shape}")
        if ret.dim() != 4:
            raise ValueError(f"检索的邻居必须是4D张量[batch_size, chunks, neighbors, neighbor_len], 但得到{ret.shape}")

        # 词嵌入
        h = self.token_emb(x)
        ret_emb = self.token_emb(ret)

        cross_attn_idx = 0
        encoder_output = None

        for layer_idx in range(len(self.self_attn_layers)):
            # 自注意力
            h = self.self_attn_layers[layer_idx](h)

            # 在第一个交叉注意力层初始化编码器输出
            if self.ca_layers and layer_idx == min(self.ca_layers):
                encoder_output = self.encoder(ret_emb, h)
                encoder_output = self.encoder_norm(encoder_output)

            # 分块交叉注意力(如果当前层需要)
            if layer_idx in self.ca_layers:
                h = self.chunked_cross_attn_layers[cross_attn_idx](h, encoder_output)
                cross_attn_idx += 1

            # 前馈网络
            h = self.ffn_layers[layer_idx](h)

        # 输出投影
        return self.output_layer(h)


def _test_retro_model():
    """RETRO模型测试函数"""
    # 基本配置
    vocab_size = 10000  # 词汇表大小
    d_model = 512  # 模型维度
    n_layers = 6  # 层数
    ca_layers = {2, 5}  # 使用交叉注意力的层
    chunk_len = 64  # 块长度
    n_heads = 8  # 注意力头数
    d_k = 64  # 每个头的维度
    d_ff = 2048  # 前馈网络隐藏层维度
    eps = 1e-6  # LayerNorm的小常数

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建编码器
    encoder = NearestNeighborEncoder(
        chunk_len=chunk_len,
        n_layers=2,
        ca_layers={1},
        d_model=d_model,
        n_heads=n_heads,
        d_k=d_k,
        d_ff=d_ff,
        eps=eps
    ).to(device)

    # 创建RETRO模型
    model = RetroModel(
        n_vocab=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        ca_layers=ca_layers,
        chunk_len=chunk_len,
        n_heads=n_heads,
        d_k=d_k,
        d_ff=d_ff,
        encoder=encoder,
        eps=eps
    ).to(device)

    # 测试不同输入长度
    for seq_len in [128, 256, 512]:
        batch_size = 4
        chunks = seq_len // chunk_len

        # 生成随机输入和检索的邻居
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        ret = torch.randint(0, vocab_size, (batch_size, chunks, 2, chunk_len), device=device)

        print(f"\n测试 seq_len={seq_len}, chunks={chunks}")

        try:
            # 前向传播
            output = model(x, ret)

            # 验证输出形状
            assert output.shape == (batch_size, seq_len, vocab_size), \
                f"输出形状错误: 期望[{batch_size}, {seq_len}, {vocab_size}], 实际{output.shape}"

            # 验证输出值
            assert not torch.isnan(output).any(), "输出包含NaN值"
            assert not torch.isinf(output).any(), "输出包含Inf值"

            print(f"测试通过! 输出形状: {output.shape}")

        except Exception as e:
            print(f"测试失败: {str(e)}")


if __name__ == '__main__':
    _test_retro_model()