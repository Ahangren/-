import math
from typing import Set
import torch
import torch.nn as nn
from anyio import sleep_until
from labml.logger import inspect
from torch.nn import functional as F


class RotaryPositionalEmbeddings(nn.Module):
    """旋转式位置编码 (RoPE)"""

    def __init__(self, d: int, base: int = 10000):
        """
        参数:
            d:     嵌入维度（必须为偶数）
            base:  频率计算的基数（控制波长范围）
        """
        super().__init__()
        assert d % 2 == 0, "嵌入维度d必须是偶数"

        # 初始化θ值：θ_i = 1/(base^(2i/d)), i ∈ [0, d//2-1]
        # 原论文中θ不可训练，这里使用register_buffer而非Parameter
        theta = 1. / (base ** (torch.arange(0, d // 2).float() / (d // 2)))
        self.register_buffer('theta', theta)  # 固定参数，不参与训练

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用旋转位置编码到输入张量

        输入:
            x:  形状为 [batch_size, seq_len, n_heads, d] 的张量
                (例如Transformer中多头注意力的Q/K/V)

        返回:
            应用旋转编码后的张量（保持原始形状）
        """
        batch_size, seq_len, n_heads, d = x.shape
        half_dim = d // 2  # 旋转编码操作需要将维度分成两半

        # 生成位置序列 [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).type_as(self.theta)

        # 计算所有位置和维度的旋转角度 mθ_i
        # 结果形状: [seq_len, d//2]
        freqs = torch.einsum('n,d->nd', positions, self.theta)

        # 计算旋转矩阵的cos和sin分量
        cos = torch.cos(freqs)[None, :, None, :]  # 广播维度 [1, seq_len, 1, d//2]
        sin = torch.sin(freqs)[None, :, None, :]  # 同上

        # 将输入x分成前半部分和后半部分
        x1, x2 = x[..., :half_dim], x[..., half_dim:]

        # 执行旋转操作（核心公式）
        # x'_i =  x_i * cos(mθ_i) - x_{i+d/2} * sin(mθ_i)
        # x'_{i+d/2} = x_i * sin(mθ_i) + x_{i+d/2} * cos(mθ_i)
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # 拼接旋转后的两部分
        return torch.cat([rotated_x1, rotated_x2], dim=-1)


# 自我注意力层（适用于因果和非因果的多头注意力）
class SelfAttention(nn.Module):
    def __init__(self,d_model:int,n_heads:int,d_k:int,is_causal:bool):

        super().__init__()

        self.is_causal=is_causal
        self.n_heads=n_heads
        self.d_model=d_model
        self.d_k=d_k
        self.scale=1/math.sqrt(d_k)

        self.query=nn.Linear(d_model,n_heads*d_k)
        self.key=nn.Linear(d_model,n_heads*d_k)
        self.value=nn.Linear(d_model,n_heads*d_k)

        self.rotaty_pe=RotaryPositionalEmbeddings(d_k)
        # self.softmax=nn.Softmax()
        self.norm=nn.LayerNorm(d_model)
        self.output=nn.Linear(n_heads*d_k,d_model)

    def mask_attention(self,attn:torch.Tensor):
        if not self.is_causal:
            return attn
        seq_len=attn.size(-1)
        mask=torch.tril(torch.ones(seq_len,seq_len,device=attn.device,dtype=torch.bool))
        return attn.masked_fill(~mask,float('-inf'))

    def forward(self,h:torch.Tensor):
        h_ser=h

        h=self.norm(h)
        batch_size, seq_len, _ = h.shape

        q=self.query(h).view(batch_size,seq_len,self.n_heads,self.d_k)
        k=self.key(h).view(batch_size,seq_len,self.n_heads,self.d_k)
        v=self.value(h).view(batch_size,seq_len,self.n_heads,self.d_k)

        # 应用旋转位置编码
        q=self.rotaty_pe(q)
        k=self.rotaty_pe(k)

        # 计算注意力
        attn=torch.einsum('bihd,bjhd->bhij',q,k)*self.scale

        # 应用因果掩码
        attn=self.mask_attention(attn)

        # 计算注意力权重
        attn=F.softmax(attn,dim=-1)


        # 计算得分
        attn=torch.einsum('bhij,bjhd->bihd',attn,v)

        # 合并多头
        out=attn.reshape(batch_size,seq_len,-1)

        # 输出投影+残差连接
        return out+h_ser

# 交叉注意力机制
class CrossAttention(nn.Module):
    def __init__(self,d_model,n_heads,d_k):
        super().__init__()
        self.d_model=d_model
        self.n_heads=n_heads
        self.d_k=d_k
        self.scale=1/math.sqrt(d_k)

        self.softmax=nn.Softmax(dim=1)
        self.query=nn.Linear(d_model,self.n_heads*self.d_k)
        self.key=nn.Linear(d_model,n_heads*d_k)
        self.value=nn.Linear(d_model,n_heads*d_k)

        self.output=nn.Linear(n_heads*d_k,d_model)
        self.norm=nn.LayerNorm(d_model)

    def forward(self,e,h):
        e_res=e

        e=self.norm(e)
        batch_size=e.shape[0]
        q=self.query(e).view(batch_size,-1,self.n_heads,self.d_k)
        k=self.key(h).view(batch_size,-1,self.n_heads,self.d_k)
        v=self.value(h).view(batch_size,-1,self.n_heads,self.d_k)

        attn=torch.einsum('bqhd,bkhd->bhqk',q,k)*self.scale

        attn=self.softmax(attn)

        out=torch.einsum('bhqk,bqhd->bqhd',attn,v)
        output=out.reshape(batch_size,-1,self.n_heads*self.d_k)


        return self.output(output+e_res)

# 分块交叉注意力机制
class ChunkedCrossAttention(nn.Module):
    def __init__(self,d_model,n_heads,d_k,chunk_len):
        super().__init__()
        self.d_model=d_model
        self.n_heads=n_heads
        self.d_k=d_k
        self.scale=1/math.sqrt(d_k)
        self.chunk_len=chunk_len

        self.query=nn.Linear(d_model,n_heads*d_k)
        self.key=nn.Linear(d_model,n_heads*d_k)
        self.value=nn.Linear(d_model,n_heads*d_k)

        self.norm=nn.LayerNorm(d_model)
        self.output=nn.Linear(n_heads*d_k,d_model)

    def forward(self,h,e):
        batch_size=h.size(0)
        chunks=e.size(1)
        if not chunks:
            return h

        h_res=h
        h=h[:,self.chunk_len-1]
        h=self.norm(h)

        if h.size(1)<chunks*self.chunk_len:
            pad_len=chunks*self.chunk_len-h.size(1)
            h=F.pad(0,0,0,pad_len)

        h=h.view(batch_size,chunks,self.chunk_len,-1)

        q=self.query(h).view(*h.shape[:-1],self.n_heads,self.d_k)
        k=self.key(e).view(*e.shape[:-1],self.n_heads,self.d_k)
        v=self.value(e).view(*e.shape[:-1],self.n_heads,self.d_k)

        attn=torch.einsum('bclhd,bcnmhd->bchlnm',q,k)*self.scale

        attn=F.softmax(attn.view(*attn.shape[:-2],-1),dim=-1)
        attn=attn.view(attn.shape[0],-1,*attn.shape[2:])

        out=torch.einsum('bchlnm,bcnmhd->bclhd',attn,v)

        out=out.reshape(batch_size,chunks*self,self.chunk_len,-1)

        out=out.reshape(batch_size,chunks*self.chunk_len,-1)
        out=self.output(out)

        out=F.pad(out,(0,0,self.chunk_len-1,0),value=0)
        return out[:,:h_res.size(1)]+h_res

# 前馈网络层
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()

        self.lin1=nn.Linear(d_model,d_ff)
        self.lin2=nn.Linear(d_ff,d_model)

        self.act=nn.ReLU()
        self.norm=nn.LayerNorm(d_model)

    def forward(self,x):
        x_ret=x
        x=self.norm(x)
        return self.lin2(self.act(self.lin1(x)))+x_ret

# 最近邻位置编码
class NearestNeighborEncoder(nn.Module):
    def __init__(self,chunk_len,n_layers,ca_layers,d_model,n_heads,d_k,d_ff):
        super().__init__()
        self.ca_calayer=ca_layers
        self.chunk_len=chunk_len
        self.ca=nn.ModuleList([
            CrossAttention(d_model,n_heads,d_k) for _ in range(len(ca_layers))
        ])

        self.attn=nn.ModuleList([
            SelfAttention(d_model,n_heads,d_k,is_causal=False) for _ in range(n_layers)
        ])

        self.ff2=nn.ModuleList([
            FeedForward(d_model,d_ff) for _ in range(n_layers)
        ])

        self.norm_h=nn.LayerNorm(d_model)

    def forward(self,e,h):
        batch_size,chunks,neighbors,neighbor_len,d_model=e.shape

        h_split=h[:,:self.chunk_len*chunks,:].reshape(batch_size,chunks,chunks,self.chunk_len,d_model)

        h_split=self.norm_h(h_split)

        p_ca=0

        for p in range(len(self.attn)):
            e=self.attn[p](e.view(-1,neighbor_len,d_model)).view(e.shape)
            if p in self.ca_layers:
                e=self.ca[p_ca](e,h_split)
                p_ca+=1
            e=self.ffw[p](e)
        return e


class RetroModel(nn.Module):
    def __init__(self,n_vocab,d_model,n_layers,ca_layers,chunk_len,n_heads,d_k,d_ff,encoder):
        super().__init__()
        self.ca_layers=ca_layers
        self.encoder=encoder

        self.emb=nn.Embedding(n_vocab,d_model)
        self.caa=nn.ModuleList([
            ChunkedCrossAttention(d_model,n_heads,d_k,chunk_len) for _ in range(len(ca_layers))
        ])

        self.attn=nn.ModuleList([
            SelfAttention(d_model,n_heads,d_k,is_causal=True) for _ in range(n_layers)
        ])

        self.ffw=nn.ModuleList([FeedForward(d_model,d_ff) for _ in range(n_layers)])

        self.read=nn.Linear(d_model,n_vocab)

        self.norm_e=nn.LayerNorm(d_model)

    def forward(self,x,ret):
        h=self.emb(x)

        ret_emb=self.emb(ret)
        p_ca=0

        for p in range(len(self.attn)):
            h=self.attn[p](h)
            if self.ca_layers and p==min(self.ca_layers):
                e=self.encoder(ret_emb,h)
                e=self.norm_e(e)

            if p in self.ca_layers:
                h=self.cca[p_ca](h,e)
                p_ca+=1
            h=self.read(h)
        return self.read(h)


def _test():
    chunk_len=4
    d_model=8
    d_ff=32
    n_heads=2
    d_k=4

    device=torch.device('cpu')
    m=RetroModel(5,d_model,6,{2,5},chunk_len,n_heads,d_k,d_ff,
                 encoder=NearestNeighborEncoder(chunk_len,2,{1},d_model,n_heads,d_k,d_ff))

    m.to(device)
    x=[1,2,3,4,0,1,2,3,4,3]
    ret = [
        [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]],
        [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]],
        ]
    res=m(torch.tensor([x]*10).to(device),torch.tensor([ret]*10).to(device))

    inspect(res)

if __name__ == '__main__':
    _test()























