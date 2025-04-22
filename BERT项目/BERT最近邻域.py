from typing import List, Optional
import faiss
import numpy as np
import torch
from labml import lab, monit
from labml_helpers.datasets.text import TextFileDataset
from labml_nn.transformers.retro.bert_embeddings import BERTChunkEmbeddings


def build_dataset(
        chunk_len: int = 16,  # 每个文本块的长度（字符数）
        batch_size: int = 64,  # BERT编码的批处理大小
        d_emb: int = 768,  # BERT嵌入维度
        n_centroids: int = 256,  # FAISS聚类中心数
        code_size: int = 64,  # FAISS压缩编码大小（建议设为64，原84可能有误）
        n_probe: int = 8,  # FAISS搜索时探查的聚类中心数
        n_train: int = 50000,  # 训练FAISS索引的样本数
        force_rebuild: bool = False  # 是否强制重新构建索引
):
    """构建文本检索数据库"""
    index_path = lab.get_data_path() / 'retro.index'

    # 如果索引已存在且不需要重建，则直接返回
    if index_path.exists() and not force_rebuild:
        print(f"Index already exists at {index_path}")
        return

    # 加载文本数据
    dataset = TextFileDataset(
        lab.get_data_path() / 'tiny_shakespeare.txt',
        list,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    )
    text = dataset.train

    # 文本分块处理（确保最后一块足够长）
    chunks = [
        text[i:i + chunk_len]
        for i in range(0, len(text) - chunk_len * 2, chunk_len)  # 确保i+chunk_len*2不越界
    ]
    chunk_offsets = np.array([i for i in range(0, len(text) - chunk_len * 2, chunk_len)])

    # BERT分块编码
    bert = BERTChunkEmbeddings(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    chunk_emb = []

    with torch.no_grad():  # 禁用梯度计算
        for i in monit.iterate('Get embeddings', range(0, len(chunks), batch_size)):
            batch = chunks[i:i + batch_size]
            embeddings = bert(batch).cpu()  # 转移到CPU避免GPU内存不足
            chunk_emb.append(embeddings)

    chunk_emb = torch.cat(chunk_emb, dim=0).numpy()  # 转换为NumPy数组供FAISS使用

    # 构建FAISS索引
    quantizer = faiss.IndexFlatL2(d_emb)
    index = faiss.IndexIVFPQ(quantizer, d_emb, n_centroids, code_size, 8)
    index.nprobe = n_probe

    # 训练索引
    if len(chunk_emb) < n_train:
        n_train = len(chunk_emb)

    with monit.section('Train index'):
        train_sample = np.random.choice(len(chunk_emb), size=n_train, replace=False)
        index.train(chunk_emb[train_sample])

    # 添加数据到索引
    with monit.section('Add to index'):
        index.add_with_ids(chunk_emb, chunk_offsets)

    # 保存索引
    with monit.section('Save'):
        faiss.write_index(index, str(index_path))
    print(f"Index saved to {index_path}")


class RetroIndex:
    def __init__(
            self,
            chunk_len: int = 16,
            n_probe: int = 8,
            n_neighbors: int = 2,
            n_extra: int = 2,
            exclude_neighbor_span: int = 8,
            device: str = 'cpu'
    ):
        """
        初始化检索器

        参数:
            device: 指定设备 ('cpu' 或 'cuda')
        """
        self.n_neighbors = n_neighbors
        self.chunk_len = chunk_len
        self.exclude_neighbor_span = exclude_neighbor_span
        self.n_extra = n_extra

        # 根据设备初始化BERT
        self.device = torch.device(device)
        self.bert = BERTChunkEmbeddings(self.device)

        # 加载FAISS索引
        index_path = lab.get_data_path() / 'retro.index'
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found at {index_path}. Run build_dataset() first.")

        with monit.section('Load index'):
            self.index = faiss.read_index(str(index_path))
            self.index.nprobe = n_probe

    def filter_neighbors(self, offset: int, neighbor_offsets: List[int]) -> List[int]:
        """过滤掉位置太近的邻居块"""
        min_offset = offset - (self.chunk_len + self.exclude_neighbor_span)
        max_offset = offset + (self.chunk_len + self.exclude_neighbor_span)
        return [n for n in neighbor_offsets if n < min_offset or n > max_offset]

    def __call__(
            self,
            query_chunks: List[str],
            offsets: Optional[List[int]] = None
    ) -> List[List[int]]:
        """
        检索相似文本块

        返回:
            每个查询块对应的邻居位置列表
        """
        # BERT编码
        with torch.no_grad():
            emb = self.bert(query_chunks).cpu().numpy()

        # FAISS搜索
        _, neighbors_offsets = self.index.search(
            emb,
            self.n_neighbors + self.n_extra
        )

        # 过滤邻近块
        if offsets is not None:
            neighbors_offsets = [
                self.filter_neighbors(off, n_off.tolist())  # 转换为list
                for off, n_off in zip(offsets, neighbors_offsets)
            ]

        # 确保返回指定数量的邻居
        return [n_off[:self.n_neighbors] for n_off in neighbors_offsets]


if __name__ == '__main__':
    # 示例用法
    build_dataset(force_rebuild=True)  # 首次运行需要构建索引

    # 初始化检索器
    retriever = RetroIndex(device='cuda' if torch.cuda.is_available() else 'cpu')

    # 测试查询
    test_queries = ["To be or not", "That is the question"]
    results = retriever(test_queries, offsets=[1000, 2000])  # 假设的偏移量
    print("Retrieved neighbor positions:", results)