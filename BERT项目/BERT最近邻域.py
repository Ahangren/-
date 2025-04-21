from typing import List,Optional
import faiss
import numpy as np
import torch

from labml import lab,monit
from labml_helpers.datasets.text import TextFileDataset
from labml_nn.transformers.retro.bert_embeddings import BERTChunkEmbeddings
from networkx import neighbors


# 建立一个支持快速检索的文本数据库，将文本分块后用BERT编码，并用FAISS建立高效索引
def build_dataset(
    chunk_len: int=16,  # 每个文本快的长度
    batch_size : int=64,  # 编码批次大小
    d_emb :int=768, # BERT嵌入维度
    n_centeroids :int=256, # FAISS聚类中心数（影响检索精度）
    code_size :int =84,  # FAISS压缩编码大小（空间与精度权衡）
    n_probe :int=8,  # FAISS搜索时探查的聚类中心数
    n_train :int=50000,  # FAISS索引的样本数
):
    # 加载本地数据
    dataset=TextFileDataset(
        lab.get_data_path()/'tiny_shakespeare.txt',
        list,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'  # 数据下载地址
    )
    text=dataset.train
    # 文本分块处理
    chunks=[
        text[i:i+chunk_len] for i in range(0,len(text),chunk_len) if i+chunk_len*2<len(text)
    ]

    chunk_offsets=np.array([i for i in range(0,len(text),chunk_len) if i+chunk_len*2<len(text)])

    n_chunks=len(chunks)
    bert=BERTChunkEmbeddings(torch.device('cpu')) # 初始化BERT嵌入器cpu模式
    chunk_emb=[]

    for i in monit.iterate('Get embeddings',range(0,n_chunks,batch_size)):
        chunk_emb.append(bert(chunks[i:i+batch_size]).cpu())
    chunk_emb=torch.cat(chunk_emb,dim=0)

    quantizer=faiss.IndexFlatL2(d_emb)
    index=faiss.IndexIVFPQ(quantizer,d_emb,n_centeroids,code_size,8)
    index.nprobe=n_probe

    random_sample=np.random.choice(
        np.arange(n_chunks),size=[min(n_train,n_chunks)],replace=True
    )

    with monit.section('Train index'):
        index.train(chunk_emb[random_sample])

    for s in monit.iterate('Index',range(0,n_chunks,1024)):
        e=min(s+1024,n_chunks)
        index.add_with_ids(chunk_emb[s:e],chunk_offsets[s:e])

    with monit.section('Save'):
        faiss.write_index(index,str(lab.get_data_path()/'retro.index'))

class RetroIndex:
    def __init__(
        self,
        chunk_len: int = 16,               # 每个文本块的长度（字符数）
        n_probe: int = 8,                  # FAISS 搜索时探查的聚类中心数
        n_neighbors: int = 2,              # 最终返回的最近邻数量
        n_extra: int = 2,                  # 初步搜索时多取的邻居数（用于后续过滤）
        exclude_neighbor_span: int = 8      # 排除邻近块的阈值（避免重叠）
    ):
        # 初始化参数
        self.n_neighbors = n_neighbors
        self.chunk_len = chunk_len
        self.exclude_neighbor_span = exclude_neighbor_span
        self.n_extra = n_extra

        # 加载 BERT 分块嵌入器（CPU模式）
        self.bert = BERTChunkEmbeddings(torch.device('cpu'))

        # 加载预训练的 FAISS 索引
        with monit.section('Load index'):
            self.index = faiss.read_index(str(lab.get_data_path() / 'retro.index'))
            self.index.nprobe = n_probe  # 设置搜索时的聚类中心探查数

    def filter_neighbors(self, offset: int, neighbor_offsets: List[int]):
        # 过滤掉与当前块位置太近的邻居块
        return [
            n for n in neighbor_offsets
            if n < offset - (self.chunk_len + self.exclude_neighbor_span)  # 邻居在查询块左侧远处
               or n > offset + (self.chunk_len + self.exclude_neighbor_span)  # 邻居在查询块右侧远处
        ]

    def __call__(self, query_chunks: List[str], offsets: Optional[List[int]]):
        # 将查询文本块通过 BERT 编码为向量
        emb = self.bert(query_chunks).cpu()

        # 在 FAISS 索引中搜索最近的邻居
        # 返回：距离（未使用）和邻居块的位置列表
        distance, neighbors_offsets = self.index.search(
            emb.numpy(),  # 查询向量
            self.n_neighbors + self.n_extra  # 搜索数量 = 最终数量 + 冗余量
        )

        # 如果提供了 offsets（查询块的位置信息），过滤掉邻近的邻居
        if offsets is not None:
            neighbors_offsets = [
                self.filter_neighbors(off, n_off)
                for off, n_off in zip(offsets, neighbors_offsets)
            ]

        # 截取前 n_neighbors 个有效邻居
        neighbors_offsets = [
            n_off[:self.n_neighbors] for n_off in neighbors_offsets
        ]

        return neighbors_offsets  # 返回最终的邻居位置列表

if __name__ == '__main__':
    build_dataset()
