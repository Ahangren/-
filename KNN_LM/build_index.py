import numpy as np
import torch
import faiss
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple


class FAISSIndexBuilder:
    """
    Build FAISS index for k-NN search with transformer embeddings

    功能:
    1. 收集Transformer模型的嵌入向量(f(c_i))和对应的token(w_i)
    2. 构建并保存FAISS索引以实现高效近邻搜索
    """

    def __init__(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                 device: torch.device, output_dir: str = './data'):
        """
        初始化索引构建器

        参数:
            model: 能够生成嵌入向量的Transformer模型
            data_loader: 提供输入批次的DataLoader
            device: 运行模型的设备(CPU/GPU)
            output_dir: 保存索引文件的目录
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 计算总token数(上下文数量)
        self.total_tokens = len(data_loader.dataset) * data_loader.batch_size - 1

        # 获取模型嵌入维度
        self.embed_dim = model.transformer.d_model if hasattr(model, 'transformer') else model.d_model

    def collect_key_values(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        收集所有(f(c_i), w_i)对并保存到内存映射文件

        返回:
            keys: 存储所有f(c_i)的numpy数组
            values: 存储所有w_i的numpy数组
        """
        # 创建内存映射文件路径
        keys_path = self.output_dir / 'keys.npy'
        vals_path = self.output_dir / 'vals.npy'

        # 初始化内存映射数组
        keys = np.memmap(keys_path, dtype=np.float32, mode='w+',
                         shape=(self.total_tokens, self.embed_dim))
        vals = np.memmap(vals_path, dtype=np.int32, mode='w+',
                         shape=(self.total_tokens, 1))

        collected = 0

        # 设置为评估模式
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="Collecting key-value pairs"):
                # 获取输入数据和目标token
                inputs = batch[0].to(self.device)
                targets = batch[1].view(-1, 1)  # 展平为[batch_size * seq_len, 1]

                # 前向传播获取嵌入
                _ = self.model(inputs)
                current_embeds = self.model.ff_input.view(-1, self.embed_dim)

                # 计算当前批次的token数
                current_batch_size = current_embeds.size(0)

                # 保存到内存映射数组
                keys[collected:collected + current_batch_size] = current_embeds.cpu().numpy()
                vals[collected:collected + current_batch_size] = targets.cpu().numpy()

                collected += current_batch_size

        return keys, vals

    def build_index(self, keys: np.ndarray, n_centroids: int = 2048,
                    code_size: int = 64, n_probe: int = 8,
                    n_train: int = 200000) -> faiss.Index:
        """
        构建FAISS索引

        参数:
            keys: 包含所有f(c_i)的numpy数组
            n_centroids: FAISS使用的聚类中心数量
            code_size: 压缩编码的大小
            n_probe: 搜索时考虑的聚类中心数
            n_train: 用于训练索引的样本数

        返回:
            构建好的FAISS索引
        """
        # 创建量化器和索引
        quantizer = faiss.IndexFlatL2(self.embed_dim)
        index = faiss.IndexIVFPQ(quantizer, self.embed_dim, n_centroids, code_size, 8)
        index.nprobe = n_probe  # 搜索时考虑的聚类中心数

        # 随机选择样本进行训练
        random_sample = np.random.choice(
            np.arange(len(keys)),
            size=[min(n_train, len(keys))],
            replace=False
        )

        # 训练索引
        with tqdm(total=1, desc="Training index") as pbar:
            index.train(keys[random_sample])
            pbar.update(1)

        # 添加所有键到索引
        with tqdm(total=len(keys), desc="Adding keys to index") as pbar:
            for start in range(0, len(keys), 1024):
                end = min(start + 1024, len(keys))
                current_keys = keys[start:end]
                current_ids = np.arange(start, end)
                index.add_with_ids(current_keys, current_ids)
                pbar.update(end - start)

        # 保存索引
        index_path = str(self.output_dir / 'faiss.index')
        faiss.write_index(index, index_path)

        return index

    def run(self):
        """执行完整的索引构建流程"""
        # 1. 收集所有键值对
        keys, vals = self.collect_key_values()

        # 2. 构建FAISS索引
        index = self.build_index(keys)

        print(f"Index built and saved to {self.output_dir}")

        return index


# 使用示例
if __name__ == "__main__":
    # 假设我们已经有一个训练好的模型和数据加载器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ...  # 你的Transformer模型
    data_loader = ...  # 你的数据加载器

    # 创建索引构建器
    index_builder = FAISSIndexBuilder(model, data_loader, device)

    # 执行索引构建
    index = index_builder.run()