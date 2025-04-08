import math
from typing import Optional

import torch
import torch.nn as nn
# from holoviews.operation.datashader import inspect

from labml.logger import inspect
from labml_nn.transformers import MultiHeadAttention
# from xlwings.pro.reports.filters import scale


# 为多头注意力机制生成一组几何衰减的斜率值
def get_slopes(n_head):
    n=2**math.floor(math.log2(n_head))
    m_0=2.0**(-8.0/n)
    m_hat=torch.pow(m_0,torch.arange(1,n+1))

    if n_head>n:
        m_1=2.0**(-4.0/n)
        m_hat_1=torch.pow(m_1,torch.arange(1,1+2*(n_head-n),2))
        m_hat=torch.cat((m_hat,m_hat_1))
    return m_hat

@torch.no_grad()
def get_alibi_biases(n_heads,mask):
    m=get_slopes(n_heads).to(mask.device)
    mask=mask.cumsum(dim=-1)
    return mask[:,:,None]*m[None,None,:]

# 覆盖多头注意力
class AlibiMultiHeadAttention(MultiHeadAttention):
    def __init__(self,heads,d_model,dropout_drop,mask=None):
        super().__init__(heads,d_model,dropout_drop)
        self.alibi_biases=None

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        assert mask is not None
        assert mask.shape[0]==mask.shape[1] and mask.shape[2]==1

        seq_len,batch_size,_=query.shape
        mask=self.prepare_mask(mask,query.shape,key.shape)

        query=self.query(query)
        key=self.key(key)
        value=self.value(value)

        scores=self.get_scores(query,key)

        scores*=self.scale

        if self.alibi_biases is None or self.alibi_biases.shape[1]<seq_len:
            self.alibi_biases=get_alibi_biases(scores.shape[-1],mask[:,:,0,0])
        scores+=self.alibi_biases[:seq_len,:seq_len,None,:]

        scores=scores.masked_fill(mask==0,float('-inf'))

        attn=self.softmax(scores)

        attn=self.dropout(attn)

        x=torch.einsum('ijbh,jbhd->ibhd',attn,value)
        x=x.reshape(seq_len,batch_size,-1)

        return self.output(x)

def _test_alibi():

    inspect(get_slopes(12).tolist(), _n=-1)
    from labml_nn.transformers.utils import subsequent_mask

    mask = subsequent_mask(8)[:, :, 0]
    inspect(mask)

    inspect(get_alibi_biases(12, mask)[:, :, 3], _n=-1)

if __name__ == '__main__':
    _test_alibi()



