# 导入模块
import math
import torch
import torch.nn as nn

from PositionalEncoding import get_positional_encoding
# from labml_nn.utils import clone_module_list

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

