# 导入相关包
import torch
import torch.nn as nn
# inspect:打印数据的一个函数
from labml.logger import  inspect
# 导入多头注意力模块
from labml_nn.transformers import MultiHeadAttention

# 数据右移函数
def shift_right(x:torch.Tensor):  # x.shape=(seq_len,batch_size,heads,d_k)
    # 创建一个第一维(从0开始)和x不一样的全为0的tensor
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])  # shape(seq_len,1,batch_size,heads,d_k)
    # inspect(zero_pad)
    # 将x和零矩阵按第一维结合
    x_padded = torch.cat([x, zero_pad], dim=1)  #shape(seq_len,batch_size+1,heads,d_k)
    # inspect(x_padded)
    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])  # shape(batch_size+1,seq_len,heads,d_k)
    # inspect(x_padded)
    x = x_padded[:-1].view_as(x)
    return x

# 重写多头注意力模块
class RelativeMultiHeadAttention(MultiHeadAttention):
    def __init__(self,heads,d_model,dropout_prob):
        super().__init__(heads,d_model,dropout_prob,bias=False)
        # 定义相对位置范围：4090
        self.P=2**12
        # 为每个头和每个可能的相对位置学习独立的嵌入向量（从-P到P），形状（2*P,heads,d_k）
        self.key_pos_embeddings=nn.Parameter(torch.zeros((self.P*2,heads,self.d_k)),requires_grad=True)
        # 为每个头和相对位置学习偏置项，增强位置感知（2*P，heads）
        self.key_pos_bias=nn.Parameter(torch.zeros((self.P*2,heads)),requires_grad=True)
        # 与查询位置无关的全局偏置，用于调整查询向量的表示（heads，d_k）
        self.query_pos_bias=nn.Parameter(torch.zeros((heads,self.d_k)),requires_grad=True)

    # 计算相对位置注意力得分矩阵
    def get_scores(self,query,key):
        # 从预定义的相对位置矩阵中切片，获取与当前位置相关的部分，[P - seq_len_k : P + seq_len_q]
        key_pos_emb=self.key_pos_embeddings[self.P-key.shape[0]:self.P+query[0]]
        # 从预定义的相对位置偏置中获取与当前位置相关的部分，[P-seq_len_k, P+seq_len_q]
        key_pos_bias=self.key_pos_bias[self.P-key.shape[0]:self.P+query.shape[0]]
        # 扩展全局偏置的维度，从（heads，d_k）->(1,1,heads,d_k)
        query_pos_bias=self.query_pos_bias[None,None,:,:]
        # 计算内容相关分数，使用查询向量加上全局偏置与键向量点积，生成基础注意力分数矩阵
        ac=torch.einsum('ibhd,jbhd->ijbh',query+query_pos_bias,key)
        # 计算查询与相对位置的点积
        b=torch.einsum('ibhd,jhd->ijbh',query,key_pos_emb)
        # 扩展相对位置维度，（1，seq_len_q+seq_len_k,1,heads）
        d=key_pos_bias[None,:,None,:]
        # 将b和d相加得到相对位置分数，传入shift_right中使数据向右移一位，获得每个词的在规定范围（P）中的前后词的相对位置分数
        bd=shift_right(b+d)
        # 将seq_len_q切去，保留seq_len_k部分，保证与ac一致。（seq_len_q，seq_len_k，heads，d_k）
        bd=bd[:,-key.shape[0]:]
        # 将内容相关分数与位置相关分数相加得到相对位置注意力
        """
        总结：
        1.通过公式，我们要计算先准备好三个参数：一个相对位置学习嵌入向量，一个相对位置学习篇偏置，一个全局偏置
        2.获取每个batch_size中的相对位置、偏置、全局偏置，
        3.通过公式：先将q与全局偏置相加后与k点积，得到内容相关分数
        4.计算位置注意力分数，
        5.将位置注意力分数和内容相关分数相加得到相对位置注意力分数
        """
        return ac+bd



def _test_shift_right():
   x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
   inspect(x)
   inspect(shift_right(x))

   x = torch.arange(1, 6)[None, :, None, None].repeat(5, 1, 1, 1)
   inspect(x[:, :, 0, 0])
   inspect(shift_right(x)[:, :, 0, 0])

   x = torch.arange(1, 6)[None, :, None, None].repeat(3, 1, 1, 1)
   inspect(x[:, :, 0, 0])
   inspect(shift_right(x)[:, :, 0, 0])

if __name__ == '__main__':

    _test_shift_right()