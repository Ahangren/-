import torch
from torch import nn

from labml.logger import inspect
from labml_nn.transformers import MultiHeadAttention

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self,d,base=10000):
        super().__init__()
        # 初始化控制频率的基数，值越大，高频分量衰减越快，一般设置为10000
        self.base=base
        # 嵌入向量维度，必须是偶数，因为旋转操作需要成对的维度
        self.d=d
        # 缓存结果，避免重复计算
        self.cos_cached=None
        self.sin_cached=None

    def _build_cache(self,x):
        # 如果缓存已经存在，且覆盖当前序列长度，则直接返回
        if self.cos_cached is not None and x.shape[0]<=self.cos_cached.shape[0]:
            return
        # 当前序列长度
        seq_len=x.shape[0]
        # 根据旋转位置编码公式计算频率分量：theta_i=1/(base^(2i/d))
        theta=1./(self.base**(torch.arange(0,self.d,2).float()/self.d)).to(x.device)
        # 序列位置索引[0,1,2,...,seq_len-1]
        seq_idx=torch.arange(seq_len,device=x.device).float().to(x.device)
        # 计算外积：pos*theta, [seq_len,d//2]
        idx_theta=torch.einsum('n,d->nd',seq_idx,theta)
        # 将theta重复一次，以便后续应用到所有维度[seq_len, d]
        idx_theta2=torch.cat([idx_theta,idx_theta],dim=1)
        # 计算余弦和正弦并且缓存，[seq_len,1,1,d]
        self.cos_cached=idx_theta2.cos()[:,None,None,:]
        self.sin_cached=idx_theta2.sin()[:,None,None,:]


    def _neg_half(self,x):

        d_2=self.d//2
        # 取反操作，先截取后半部分的数据并且取负，在取前半部分的数据后将两个数据切合
        return torch.cat([-x[:,:,:,d_2:],x[:,:,:,:d_2]],dim=-1)

    def forward(self,x):
        # 构建sin和con缓存
        self._build_cache(x)
        # 分割输入，如果x的大小大于我们要嵌入的维度d，则把大于的部分舍弃不参与旋转计算
        x_rope,x_pass=x[...,:self.d],x[...,self.d:]
        # 旋转后半部分
        neg_half_x=self._neg_half(x_rope)
        # 应用旋转位置编码，
        x_rope=(x_rope*self.cos_cached[:x.shape[0]])+(neg_half_x*self.sin_cached[:x.shape[0]])

        return torch.cat((x_rope,x_pass),dim=-1)

# 使用旋转位置编码实现多头注意力模型
class RotaryPEMultiHeadAttention(MultiHeadAttention):
    def __init__(self,heads,d_model,rope_percentage,dropout_prob):
        # rope_percentage: float类型，表示使用多少的维度参与旋转位置编码
        super().__init__(heads,d_model,dropout_prob)
        # 每个头有多少数据参与旋转位置编码
        d_rope=int(self.d_k*rope_percentage)
        # Query和Key分别创建独立的RoPE编码器
        self.query_rotary_pe=RotaryPositionalEmbeddings(d_rope)
        self.key_rotary_pe=RotaryPositionalEmbeddings(d_rope)

    def get_scores(self,query,key):
        # 计算q和k之间的加上旋转位置编码的注意力分数
        return torch.einsum('ibhd,jbhd->ijbh',self.query_rotary_pe(query),self.key_rotary_pe(query))

def _test_rotary():
    x=torch.tensor([[1,2,3,4],[4,5,6,7],[7,8,9,10]])
    x=x[:,None,None,:]
    inspect(x)

    rotary_pe=RotaryPositionalEmbeddings(4)
    inspect(rotary_pe(x))

if __name__ == '__main__':
    _test_rotary()