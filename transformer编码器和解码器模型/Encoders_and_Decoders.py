# 导入模块
import math
import torch
import torch.nn as nn

from labml_nn.utils import clone_module_list
from feed_forward import FeedForward
from 多头注意力模型.Polycephalic_autoattention import MultiHeadAttention
from PositionalEncoding import get_positional_encoding

# 位置编码层
class EmbeddingsWithPositionalEncoding(nn.Module):
    def __init__(self,d_model,n_vocab,max_len=5000):
        super().__init__()
        self.linear=nn.Embedding(n_vocab,d_model)
        self.d_model=d_model
        self.register_buffer('positional_encoding',get_positional_encoding(d_model, max_len))

    def forward(self,x):
        pe=self.positional_encoding[:x.shape[0]].requires_grad_(False)
        return self.linear(x)*math.sqrt(self.d_model)+pe

# Transformer Layer层
class TransformerLayer:
    def __init__(self,d_model,self_attn,src_attn,feed_forward,dropout_prob):
        super().__init__()
        # 初始化模型维度大小
        self.size=d_model
        # 初始化自注意模型
        self.self_attn=self_attn
        # 初始化源注意力模型
        self.src_attn=src_attn
        # 初始化前馈神经网络层
        self.feed_forward=feed_forward
        # 初始化Dropout
        self.dropout=nn.Dropout(dropout_prob)
        # 初始化自注意力层的归一化层
        self.norm_self_attn=nn.LayerNorm([d_model])
        # 判断是否为源注意力模型，其实就是判断是不是解码器
        if self.src_attn is not None:
            # 如果是解码器就初始化解码器层的归一化层
            self.norm_src_attn=nn.LayerNorm([d_model])
        # 初始化前馈神经网络层的归一化层
        self.norm_ff=nn.LayerNorm([d_model])
        # 判断是否保存模型到前馈网络层
        self.is_save_ff_input=False

    def forward(self,x,mask=None,src=None,src_mask=None):
        """
        模型前向传播结构为：
        1.归一化层
        2.自注意力层
        4.残差连接和dropout
        5.如果是解码器的话这里还要进行一个源注意力层，如果是编码器的话则没有
        6.前馈网络前的归一化层
        7.前馈神经网络层
        8.残差连接后输出
        """
        # 在进行自注意力计算之前先进行归一化
        z=self.norm_self_attn(x)
        # 自注意力层
        self_attn=self.self_attn(query=z,key=z,value=z,mask=mask)
        # 残差连接+dropout层
        x=x+self.dropout(self_attn)

        if src is not None:
            # 判断是否为解码器，如果是先归一化
            z=self.norm_src_attn(x)
            # 在进行源注意力层，这里只有q来自解码器，k和v来自编码器
            attn_src=self.src_attn(query=z,key=src,value=src,mask=src_mask)
            # 残差连接
            x=x+self.dropout(attn_src)
        # 前馈神经网络前先归一化
        z=self.norm_ff(x)
        # 判断是否要将模型保存到前馈网络
        if self.is_save_ff_input:
            self.ff_input=z.clpne()
        # 前馈网络层
        ff=self.feed_forward(z)
        # 最后残差连接
        x=x+self.dropout(ff)

        return x

class Encoder(nn.Module):
    def __init__(self,layer,n_layers):
        super().__init__()
        # layer就是上面的TransformerLayer，这里的意思是复制多个将前一个的输出作为后一个的输入
        self.layers=clone_module_list(layer,n_layers)
        # 归一化层
        self.norm=nn.LayerNorm([layer.size])

    def forward(self,x,mask):

        for layer in self.layers:
            x=layer(x,mask)
        x=self.norm(x)
        return x

# 解码器模块
class Decoder(nn.Module):
    def __init__(self,layer,n_layers):
        super().__init__()
        self.layers=clone_module_list(layer,n_layers)
        self.norm=nn.LayerNorm([layer.size])

    def forward(self,x,memory,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x=x,mask=tgt_mask,src=memory,src_mask=src_mask)

        return self.norm(x)

# 生成器模块
class Generator(nn.Module):
    def __init__(self,n_vocab,d_model):
        super().__init__()
        self.projection=nn.Linear(d_model,n_vocab)

    def forward(self,x):
        return self.projection(x)

# 组合编码器和解码器模块
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generate):
        super().__init__()
        # 初始化编码器
        self.encoder=encoder
        # 初始化解码器
        self.decoder=decoder
        # 编码器的词嵌入
        self.src_embed=src_embed
        # 解码器的词嵌入
        self.tgt_embed=tgt_embed
        # 初始化生成器
        self.generate=generate

        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

    def forward(self,src,tgt,src_mask,tgt_mask):
        enc=self.encode(src,src_mask)

        return self.decode(enc,src_mask,tgt,tgt_mask)

    def encode(self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)


    def decode(self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)


