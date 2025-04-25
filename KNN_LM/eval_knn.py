import torch
import faiss
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm


class KNNEvaluator:
    """
    评估k近邻增强的Transformer语言模型

    主要功能：
    1. 结合Transformer预测和k近邻检索结果
    2. 支持多种模型和k-NN的权重组合方案
    3. 基于FAISS的高效最近邻搜索
    """

    def __init__(self, model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 device: torch.device,
                 index_path: str = './data/faiss.index',
                 keys_path: str = './data/keys.npy',
                 vals_path: str = './data/vals.npy'):
        """
        初始化评估器

        参数:
            model: Transformer语言模型
            data_loader: 验证数据加载器
            device: 运行评估的设备
            index_path: FAISS索引保存路径
            keys_path: 键(嵌入向量)保存路径
            vals_path: 值(token ID)保存路径
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device

        # 加载预构建的索引和数据
        self.index, self.keys_store, self.vals_store = self._load_index(
            index_path, keys_path, vals_path)

        # 设置模型为评估模式
        self.model.eval()

    def _load_index(self, index_path: str, keys_path: str, vals_path: str) -> Tuple:
        """加载FAISS索引和内存映射数组"""
        # 加载FAISS索引
        index = faiss.read_index(index_path)
        index.nprobe = 8  # 搜索时探查的聚类中心数量

        # 加载内存映射数组
        keys_store = np.memmap(keys_path, dtype=np.float32, mode='r')
        vals_store = np.memmap(vals_path, dtype=np.int32, mode='r')

        return index, keys_store, vals_store

    def knn_search(self, queries: torch.Tensor, n_tokens: int, k: int = 10) -> torch.Tensor:
        """
        执行k近邻搜索并计算token概率

        参数:
            queries: 查询嵌入向量 [batch, seq_len, embed_dim]
            n_tokens: 词汇表大小
            k: 最近邻数量

        返回:
            来自k-NN的token逻辑值 [batch, seq_len, vocab_size]
        """
        original_shape = queries.shape
        queries = queries.view(-1, original_shape[-1]).cpu()

        # 使用FAISS索引搜索
        distances, indices = self.index.search(queries.numpy(), k)

        # 获取最近的键和值
        nearest_keys = torch.from_numpy(self.keys_store[indices]).to(queries.device)
        nearest_vals = torch.from_numpy(self.vals_store[indices]).squeeze(-1)

        # 计算余弦相似度
        queries_n = queries / (queries.norm(dim=-1, keepdim=True) + 1e-10)
        keys_n = nearest_keys / (nearest_keys.norm(dim=-1, keepdim=True) + 1e-10)
        similarities = (keys_n * queries_n.unsqueeze(1)).sum(dim=-1)

        # 创建token逻辑值
        logits = torch.zeros(queries.shape[0], n_tokens,
                             dtype=similarities.dtype,
                             device=queries.device)

        # 将相似度分散到token位置
        logits.scatter_(1, nearest_vals, similarities, reduce='add')

        return logits.view(*original_shape[:-1], n_tokens)

    def evaluate(self, knn_weights: List[float],
                 last_n: Optional[int] = None) -> List[float]:
        """
        使用不同的k-NN权重因子评估模型

        参数:
            knn_weights: k-NN权重列表(0=仅使用Transformer, 1=仅使用k-NN)
            last_n: 仅评估每个序列的最后n个token

        返回:
            每个权重对应的困惑度列表
        """
        losses = [[] for _ in knn_weights]
        n_samples = []

        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="Evaluating"):
                data, targets = batch[0].to(self.device), batch[1].to(self.device)

                # 获取模型预测结果
                model_out, _ = self.model(data)

                # 获取k-NN预测结果
                knn_out = self.knn_search(self.model.ff_input,
                                          n_tokens=self.model.generator.proj.out_features)
                knn_out = knn_out.to(self.device)

                # 可选：仅使用最后n个token
                if last_n:
                    model_out = model_out[-last_n:]
                    knn_out = knn_out[-last_n:]
                    targets = targets[-last_n:]

                # 为每个权重计算损失
                batch_samples = model_out.shape[0] * model_out.shape[1]
                n_samples.append(batch_samples)

                for i, weight in enumerate(knn_weights):
                    combined = weight * knn_out + (1 - weight) * model_out
                    loss = torch.nn.functional.cross_entropy(
                        combined.view(-1, combined.size(-1)),
                        targets.view(-1),
                        reduction='none'
                    ).mean()
                    losses[i].append(loss.item() * batch_samples)

        # 计算每个权重的平均损失
        total_samples = sum(n_samples)
        return [sum(loss) / total_samples for loss in losses]


# 使用示例
if __name__ == "__main__":
    # 初始化组件(替换为实际实现)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ...  # 训练好的Transformer模型
    val_loader = ...  # 验证数据加载器

    # 创建评估器
    evaluator = KNNEvaluator(
        model=model,
        data_loader=val_loader,
        device=device,
        index_path='./data/faiss.index',
        keys_path='./data/keys.npy',
        vals_path='./data/vals.npy'
    )

    # 使用不同的k-NN权重评估
    weights = [i / 10 for i in range(11)]  # [0.0, 0.1, ..., 1.0]
    perplexities = evaluator.evaluate(knn_weights=weights)

    # 打印结果
    for w, ppl in zip(weights, perplexities):
        print(f"k-NN weight: {w:.1f} | Perplexity: {ppl:.2f}")