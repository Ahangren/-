import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.optimizers.configs import OptimizerConfigs
from labml_nn.transformers import TransformerConfigs,Encoder
from labml_nn.transformers.utils import subsequent_mask


class GPT(Module):
    """
    GPT模型
    这个模型由三部分组成：1.词嵌入层，（包含位置编码）。2.Transformer编码器，3.输出词概率的线性层
    这是GPT模型的核心架构，采用Transformer的Decoder-only结构
    通过自回归方式生成文本，每次只能看到位置之前的token
    """
    def __init__(self,encoder:Encoder,src_embed:Module,generator:Module):
        """
        初始化参数
        :param encoder: Transformer编码器模块
        :param src_embed: 包含位置编码的词嵌入模块
        :param generator: 最后的全连接层，用于输出词概率
        """
        super().__init__()
        self.src_embed=src_embed  # 词嵌入+位置编码
        self.encoder=encoder  # Transformer编码器
        self.generator=generator  # 输出层
        # 掩码会在第一次前向传播时初始化
        self.make=None

    def forward(self,x:torch.Tensor):
        # 如果掩码未初始化或大小不匹配，创建新的后续掩码
        # 这种延迟初始化可以适用不同的长度输入
        if self.mask is None or self.mask.size(0) !=len(x):
            # 后续掩码，防止看到未来信息
            self.mask=subsequent_mask(len(x)).to(x.device)
        # 获取带有位置编码的词嵌入
        x=self.src_embed(x)
        # Transformer编码器处理
        x=self.encoder(x,self.mask)
        # 获得输出概率
        x=self.generator(x)
        return x,None

class Configs(NLPAutoRegressionConfigs):
    """
    配置类
    继承自NLPAutoRegressionConfigs，用于管理GPT模型的所有配置参数
    这种配置类，设计使得超参数管理更加结构化，便于实验管理
    """
    # GPT模型示例
    model:GPT
    # Transformer配置
    transformers:TransformerConfigs
    # 权重衰减系数
    weight_decay:float=0.1
    # 预热步数（用于学习率调度）
    warmup_steps:float=0.1

    # 使用自定义优化器
    optimizer = 'transformer_optimizer'

@option(Configs.transformer,'GPT')
def _transformer_configs(c:Configs):
    """
    这里配置了GPU特有的Transformer参数，
    特别是使用了GELU激活函数，这是GPU的标注性选择
    """
    # 使用可配置的Transformer实现
    conf=TransformerConfigs()
    # 设置词表大小（嵌入层和输出层）
    conf.n_src_vocab=c.n_tokens
    conf.n_tgt_vocab=c.n_tokens
    # GPT使用GELU作为前馈网络的激活函数
    conf.ffn.activation='GELU'

    return conf


def _init_weights(module):
    """
    权重初始化
    线性层和嵌入层的权重初始化为N(0，0.02)的正太分布，而不是默认的Xavier初始化
    这种初始化方式是GPT模型的另外一个特点，较小的标准差有助于训练稳定性
    """
    if not isinstance(module,(nn.Linear,nn.Embedding)):
        return

    module.weight.data.normal_(mean=0.0,std=0.02)

    # 如果由偏置项，初始化为0
    if isinstance(module,nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

@option(Configs.model)
def _model(c:Configs):
    """
    创建GPU模型并且初始化权重
    这是工厂函数模式，通过配置动态创建模型示例
    """
    m=GPT(c.transformers.encoder,
          c.transformers.src_embed,
          c.transformers.generator).to(c.device)

    # 应用自定义权重初始化
    m.apply(_init_weights)

    return m


@option(NLPAutoRegressionConfigs.optimizer)
def transformer_optimizer(c:NLPAutoRegressionConfigs):
    """
    创建带权重衰减的自定义优化器
    代码参考自minGPT项目，只对线性层的权重应用权重衰减
    这种精细化的权重衰减策略可以防止过拟合，同时不影响其他参数（如LayerNorm参数）的学习
    """
    # 收集需要应用权重衰减的参数名
    decay=set()
    for mn,m in c.model.named_modules():  # 遍历c中的所有模块
        for pn,p in m.named_parameters():  # 遍历模块的所有参数
            fpn=f'{mn}.{pn}' if mn else pn  # 完整参数名（如"transformer.layers.0.linear1.weight"）
            # 如果是线性层的weight参数，加入衰减集合
            if fpn.endswith('weight') and isinstance(m,nn.Linear):
                decay.add(fpn)

    # 获取所有参数
    param_dict={pn:p for pn ,p in c.model.named_parameters()}
    # 不需要衰减的参数=所有的参数-需要衰减的参数
    no_decay=set(param_dict.keys())-decay
    # 创建两个参数组
    opt_groups=[
        {'params':[param_dict[pn] for pn in sorted(list(decay))] ,"weight_decay":c.weight_decay}, # 衰减组
        {"params":[param_dict[pn] for pn in sorted(list(no_decay))],"weight_decay":0.0} # 不衰减组
    ]

    # 创建可配置的优化器
    optimizer=OptimizerConfigs()
    # 设置参数组
    optimizer.parameters=opt_groups

    # 使用AdamW优化器+余弦退火学习率调度
    optimizer.optimizer='AdamWarmupCosineDecay'

    # 设置模型维度（影响学习率计算）
    optimizer.d_model=c.d_model

    # 基础学习率（GPU使用6e-4）
    optimizer.learning_rate=6e-4

    # Adam的超参数
    optimizer.betas=(0.9,0.95)  # (beta1,bate2)

    optimizer.eps=1e-8  # 数值稳定项

    # 解耦权重衰减（AdamW的关键改进）
    optimizer.weight_decouple=True

    # 学习率调度相关
    optimizer.total_steps = c.epochs * len(c.text.train) // (c.batch_size * c.seq_len)  # 总训练步数
    optimizer.warmup = c.warmup_steps // (c.batch_size * c.seq_len)  # 预热步数









