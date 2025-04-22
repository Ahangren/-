"""
---
title: 压缩Transformer实验
summary: 这个实验在tiny Shakespeare数据集上训练压缩Transformer模型
---

# 压缩Transformer实验

这是一个带注释的PyTorch实验，用于训练压缩Transformer模型。
"""
from typing import List, Tuple, NamedTuple

import torch
import torch.nn as nn

from labml import experiment, tracker, monit, logger
from labml.configs import option
from labml.logger import Text
from labml_helpers.metrics.simple_state import SimpleStateModule
from labml_helpers.module import Module
from labml_helpers.train_valid import BatchIndex, hook_model_outputs
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.transformers.compressive import CompressiveTransformer, AttentionReconstructionLoss, \
    CompressiveTransformerLayer, Conv1dCompression


class CompressedMemory(NamedTuple):
    """
    ## 压缩记忆容器
    🧠 使用命名元组存储两种记忆：
    - mem: 主记忆（短期记忆）
    - c_mem: 压缩记忆（长期记忆）
    """
    mem: List[torch.Tensor]
    c_mem: List[torch.Tensor]


class AutoregressiveModel(Module):
    """
    ## 自回归模型
    🧠 核心模型结构，包含嵌入层、Transformer和解码器
    """

    def __init__(self, n_vocab: int, d_model: int, transformer: CompressiveTransformer):
        super().__init__()
        # 词嵌入层
        self.src_embed = nn.Embedding(n_vocab, d_model)
        # 压缩Transformer模块
        self.transformer = transformer
        # 输出生成层
        self.generator = nn.Linear(d_model, n_vocab)
        # 注意力掩码缓存
        self.mask_x = None  # 序列自身的掩码
        self.mask_mem = None  # 记忆部分的掩码

    def forward(self, x: torch.Tensor, mem: CompressedMemory):
        """
        🧠 前向传播流程：
        1. 处理记忆输入
        2. 生成注意力掩码
        3. 词嵌入
        4. 通过Transformer
        5. 生成输出
        """
        # 获取记忆和压缩记忆
        if mem is not None:
            mem, c_mem = mem.mem, mem.c_mem
        else:
            mem = []
            c_mem = []

        # 计算记忆总长度（用于掩码生成）
        m_len = len(mem[0]) if mem else 0
        if c_mem:
            m_len += len(c_mem[0])

        # 生成序列的因果掩码（防止看到未来信息）
        if self.mask_x is None or self.mask_x.shape[0] < len(x):
            from labml_nn.transformers.utils import subsequent_mask
            self.mask_x = subsequent_mask(len(x)).to(x.device)
        # 生成记忆部分的掩码（全1表示完全可见）
        if self.mask_mem is None or self.mask_mem.shape[1] < m_len or self.mask_mem.shape[0] < len(x):
            self.mask_mem = self.mask_x.new_ones(len(x), m_len, 1)

        # 合并记忆掩码和序列掩码
        if m_len:
            mask = torch.cat((self.mask_mem[:len(x), :m_len], self.mask_x[:len(x), :len(x)]), dim=1)
        else:
            mask = self.mask_x[:len(x), :len(x)]

        # 词嵌入
        x = self.src_embed(x)
        # 通过Transformer
        res, mem = self.transformer(x, mem, c_mem, mask)
        # 生成下一个token的logits
        res = self.generator(res)

        return res, mem


class Configs(NLPAutoRegressionConfigs):
    """
    ## 实验配置
    🧠 包含模型超参数和训练设置
    """

    model: AutoregressiveModel

    # 模型维度
    d_model: int = 128
    # 注意力头数
    heads: int = 4
    # Dropout概率
    dropout: float = 0.0
    # 前馈层中间维度
    d_ff: int = 256
    # Transformer层数
    n_layers: int = 6
    # 记忆长度
    mem_len: int = 8
    # 记忆状态管理模块
    memory = SimpleStateModule()
    # 注意力重建损失
    attention_reconstruction_loss: AttentionReconstructionLoss
    # 压缩率（每隔多少步压缩一次）
    compression_rate: int = 4
    # 压缩记忆长度
    c_mem_len: int = 128

    def init(self):
        """
        🧠 初始化跟踪器和状态模块
        """
        # 配置跟踪指标
        tracker.set_scalar("accuracy.*", True)
        tracker.set_scalar("loss.*", True)
        # 不在终端显示注意力重建损失
        tracker.set_scalar("ar_loss.*", False)
        # 添加钩子记录模型输出
        hook_model_outputs(self.mode, self.model, 'model')
        # 保持训练和验证的准确率和记忆状态分离
        self.state_modules = [self.accuracy, self.memory]

    @torch.no_grad()
    def merge_compress_memory(self, mem: CompressedMemory, new_mem: List[torch.Tensor]) \
            -> Tuple[CompressedMemory, List[torch.Tensor]]:
        """
        ## 合并和压缩记忆
        🧠 核心记忆管理逻辑：
        1. 合并新记忆
        2. 检查是否超限
        3. 压缩旧记忆
        4. 维护压缩记忆队列
        """

        # 如果配置为不使用记忆
        if self.mem_len == 0 and self.c_mem_len == 0:
            return CompressedMemory([], []), []

        # 解构记忆
        if mem is not None:
            mem, c_mem = mem.mem, mem.c_mem
        else:
            mem, c_mem = [], []

        # 合并新记忆到主记忆
        if mem:
            mem = [torch.cat((m, x), dim=0) for m, x in zip(mem, new_mem)]
        else:
            mem = new_mem

        # 如果主记忆超过限制长度
        if len(mem[0]) > self.mem_len:
            # 计算需要压缩的记忆块数
            n_c_mem = (len(mem[0]) - self.mem_len + self.compression_rate - 1) // self.compression_rate
            # 计算实际要压缩的记忆长度
            n_old = n_c_mem * self.compression_rate

            # 待压缩的记忆
            mem_to_compress = []
            # 不压缩的记忆
            uncompressed_mem = []

            # 分割记忆
            for m in mem:
                cm, m = torch.split(m, [n_old, len(m) - n_old])
                mem_to_compress.append(cm)
                uncompressed_mem.append(m)
            mem = uncompressed_mem

            # 压缩记忆
            new_c_mem = []
            for i, layer in enumerate(self.model.transformer.layers):
                new_c_mem.append(layer.compress(mem_to_compress[i]))

            # 合并新旧压缩记忆
            if c_mem:
                c_mem = [torch.cat((m, nm), dim=0) for m, nm in zip(c_mem, new_c_mem)]
            else:
                c_mem = new_c_mem

            # 压缩记忆长度限制
            if len(c_mem[0]) > self.c_mem_len:
                c_mem = [m[-self.c_mem_len:] for m in c_mem]
        else:
            mem_to_compress = []

        return CompressedMemory(mem, c_mem), mem_to_compress

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        ## 训练/验证步骤
        🧠 单批次处理流程：
        1. 数据准备
        2. 记忆处理
        3. 损失计算
        4. 反向传播（训练模式）
        """

        # 数据移动到设备
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # 训练模式下更新全局步数
        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        # 模型前向传播
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # 获取记忆
            mem = self.memory.get()
            # 模型推理
            output, new_mem = self.model(data, mem)
            # 合并压缩记忆
            mem, mem_to_compress = self.merge_compress_memory(mem, new_mem)
            # 更新记忆
            self.memory.set(mem)

        # 计算交叉熵损失
        loss = self.loss_func(output, target)
        tracker.add("loss.", loss)

        # 如果有记忆被压缩，计算重建损失
        if mem_to_compress:
            ar_loss = self.attention_reconstruction_loss(new_mem, mem_to_compress)
            tracker.add("ar_loss.", ar_loss)
            loss = loss + ar_loss  # 总损失

        # 计算准确率
        self.accuracy(output, target)
        self.accuracy.track()

        # 训练模式下的反向传播
        if self.mode.is_train:
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            self.optimizer.step()
            # 每个epoch最后记录模型状态
            if batch_idx.is_last:
                tracker.add('model', self.model)
            self.optimizer.zero_grad()  # 清空梯度

        tracker.save()  # 保存指标

    def sample(self):
        """
        ## 生成样本
        🧠 训练过程中定期生成文本示例
        """

        # 起始提示
        prompt = self.prompt
        # 收集输出用于打印
        log = [(prompt, Text.subtle)]
        # 初始化记忆
        mem = CompressedMemory([], [])
        # 生成25个token
        for i in monit.iterate('Sample', 25):
            # 文本转token
            data = self.text.text_to_i(prompt).unsqueeze(-1)
            data = data.to(self.device)
            # 模型预测
            output, new_mem = self.model(data, mem)
            # 贪心解码
            output = output.argmax(dim=-1).squeeze(1)
            # 更新提示文本
            prompt += self.prompt_separator + self.text.itos[output[-1]]
            # 下次迭代只使用最后一个字符
            prompt = prompt[-1:]
            # 记录输出
            log += [(self.prompt_separator + self.text.itos[output[-1]], Text.value)]
            # 更新记忆
            mem, _ = self.merge_compress_memory(mem, new_mem)

        logger.log(log)  # 打印生成结果


@option(Configs.model)
def autoregressive_model(c: Configs):
    """
    ## 初始化自回归模型
    🧠 创建包含相对位置编码的压缩Transformer
    """
    from labml_nn.transformers.xl import RelativeMultiHeadAttention
    from labml_nn.transformers.feed_forward import FeedForward
    m = AutoregressiveModel(c.n_tokens, c.d_model, CompressiveTransformer(
        CompressiveTransformerLayer(d_model=c.d_model,
                                    self_attn=RelativeMultiHeadAttention(c.heads, c.d_model, c.dropout),
                                    feed_forward=FeedForward(c.d_model, c.d_ff, c.dropout),
                                    dropout_prob=c.dropout,
                                    compress=Conv1dCompression(c.compression_rate, c.d_model)), c.n_layers))
    return m.to(c.device)


@option(Configs.attention_reconstruction_loss)
def attention_reconstruction_loss(c: Configs):
    """
    ## 初始化注意力重建损失
    🧠 确保压缩后的记忆能保留原始信息
    """
    return AttentionReconstructionLoss(c.model.transformer.layers)


def main():
    """
    ## 运行实验
    🧠 主函数流程：
    1. 创建实验
    2. 加载配置
    3. 启动训练
    """
    # 创建实验
    experiment.create(name="compressive_transformer", comment='')
    # 创建配置
    conf = Configs()
    # 加载配置
    experiment.configs(conf,
                       {'tokenizer': 'character',
                        'text': 'tiny_shakespeare',
                        'optimizer.learning_rate': 2.5e-4,
                        'optimizer.optimizer': 'AdamW',
                        'prompt': 'It is',
                        'prompt_separator': '',

                        'train_loader': 'sequential_train_loader',
                        'valid_loader': 'sequential_valid_loader',

                        'seq_len': 8,
                        'mem_len': 8,
                        'epochs': 128,
                        'batch_size': 32,
                        'inner_iterations': 25,
                        'compression_rate': 2,
                        })

    # 设置模型保存
    experiment.add_pytorch_models({'model': conf.model})

    # 启动实验
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()