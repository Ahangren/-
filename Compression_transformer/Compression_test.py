from typing import List,Tuple,NamedTuple

import  torch
import torch.nn as nn

from labml import monit,experiment,tracker,logger
from labml.configs import option
from labml.logger import Text
from labml_helpers.metrics.simple_state import SimpleStateModule
from labml_helpers.module import Module
from labml_helpers.train_valid import BatchIndex,hook_model_outputs
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.transformers.compressive import CompressiveTransformer, AttentionReconstructionLoss, \
    CompressiveTransformerLayer, Conv1dCompression
from snappy import uncompress
from statsmodels.sandbox.distributions.examples.matchdist import targetdist


# 封装模型的记忆状态
class CompressedMemory(NamedTuple):
    mem:List[torch.Tensor]   # 主记忆（存储历史序列的压缩表示）
    c_mem:List[torch.TEnsor]  # 压缩记忆（进一步压缩的长期记忆）


class AutoregressiveModel(Module):
    def __init__(self,n_vocab:int,d_model:int,transform:CompressiveTransformer):
        super().__init__()
        self.src_embd=nn.Embedding(n_vocab,d_model)  # 词嵌入层,把单词变成数字密码
        self.transform=transform   # 压缩transform模块 ，处理信息的核心大脑
        self.generator=nn.Linear(d_model,n_vocab)  # 输出层，输出词的生成概率，把大脑中的输出变成答案

        self.mask_x=None  # 序列自身的注意力掩码（防止未来信息泄露）下三角矩阵
        self.mask_mem=None  # 记忆部分的注意力掩码  全1矩阵

    def forward(self,x:torch.Tensor,mem:CompressedMemory):
        if mem is not None:  # 检查是否有记忆
            mem,c_mem=mem.mem,mem.c_mem  # 打开记忆盒子
        else:
            mem=[]  # 初始化记忆（比如新学生对这里是没有记忆的）
            c_mem=[]

        # mask_x的shape（当前序列长度，当前序列长度）
        # mask_mem的shape（当前序列长度，记忆长度）
        m_len=mem[0].shape[1] if mem else 0  # 记录记忆长度

        if self.mask_x is None or self.mask_x.shape[0]<len(x): # 如果当前掩码序列不够
            from labml_nn.transformers.utils import subsequent_mask
            self.mask_x=subsequent_mask(len(x)).to(x.device)  # 重新生成掩码（当前序列长度，当前序列长度）

        if self.mask_mem is None or self.mask_mem.shape[1]<m_len or self.mask_mem.shape[0]<len(x): # 如果记忆掩码长度不够
            self.mask_mem=self.mask_x.new_ones(len(x),m_len,1)  # 重新生成全1矩阵

        if m_len:
            # 拼接掩码矩阵，保证能看到记忆中的单词和当前的单词
            mask=torch.cat((self.mask_mem[:len(x),:m_len],self.mask_x[:len(x),:len(x)]),dim=1)
        else:
            # 没有记忆时间直接使当前自身的掩码（当前序列长度，当前序列长度）
            mask=self.mask_x[:len(x),:len(x)]
        # 将单词变成数字序列
        x=self.src_embd(x)
        # 开始思考
        res,mem=self.transform(x,mem,c_mem,mask)
        # 输出答案
        res=self.generator(res)

        return res,CompressedMemory(mem, c_mem)

# 配置模块
class Configs(NLPAutoRegressionConfigs):
    model:AutoregressiveModel # 自回归模型
    d_model:int=128  # 隐藏层数量
    heads:int=4  # Transformer中的多头数
    dropout:float=0.0  # dropout概率
    d_ff:int=256  #  FeedForward中的中间层数量
    n_layers:int=6  # Transformer层数
    mem_len:int=8  # 主记忆最大长度
    memory=SimpleStateModule()  # 记忆状态管理模块
    attention_reconstruction_loss:AttentionReconstructionLoss  # 注意力重建损失
    compression_rate:int=4  # 记忆压缩概率（每隔多少步压缩一次）
    c_mem_len:int=128  # 压缩记忆的最大长度

    def init(self):
        tracker.set_scalar('ar_loss.*',True)  # 跟踪回归损失
        tracker.set_scalar('loss.*',True) # 跟踪总损失
        tracker.set_scalar('ar_loss.*',False) # 不打印日志输出

        hook_model_outputs(self.model,self.model,'model')  # 钩子函数监控模型输出
        self.state_modules=[self.accuracy,self.memory]  # 状态管理模块
    # 合并新记忆和老记忆
    @torch.no_grad()
    def merge_compress_memory(self,
                              mem:CompressedMemory,new_mem:List[torch.Tensor])->Tuple[CompressedMemory,List[torch.Tensor]]:
        if self.mem_len==0 and self.c_mem_len==0:
            # 如果mem_len和c_mem_len都是0表示不启用记忆，返回空
            return CompressedMemory([],[]),[]

        # 初始化记忆
        if mem:
            mem,c_mem=mem.mem,mem.c_mem
        else:
            mem=[]
            c_mem=[]

        # 合并新记忆
        if mem:
            # 如果有以前的记忆，讲新的记忆拼接在老记忆的后面
            mem=[torch.cat((m,x),dim=0) for m,x in zip(mem,new_mem)]
        else:
            mem=new_mem

        # 如果记忆的长度大于了主记忆的最大长度
        if len(mem[0])>self.mem_len:
            # 计算需要压缩的记忆块的数量
            n_c_mem=(len(mem[0])-self.mem_len+self.compression_rate-1)//self.compression_rate
            # 计算需要压缩的记忆长度（compression=4，表示没四步压缩一次）
            n_old=n_c_mem*self.compression_rate

            # 分割记忆
            mem_to_compress=[]  #带压缩的记忆
            uncompress_mem=[] # 不压缩的记忆

            for m in mem:
                cm,m=torch.split(m,[n_old,len(m)-n_old])
                mem_to_compress.append(cm)
                uncompress_mem.append(m)
            mem=uncompress_mem # 更新主记忆

            # 压缩记忆
            new_c_mem=[]

            for i,layer in enumerate(self.model.transform.layers):
                new_c_mem.append(layer.compress(mem_to_compress[i]))  # 调用压缩函数

            if c_mem:
                c_mem=[torch.cat((m,nm),dim=0) for m ,nm in zip(c_mem,new_c_mem)]
            else:
                c_mem=new_c_mem

            if len(c_mem[0])>self.c_mem_len:
                c_mem=[m[-self.c_mem_len:] for m in c_mem]
        else:
            mem_to_compress=[]

        return CompressedMemory(mem, c_mem),mem_to_compress

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        负责单批次训练/验证的核心逻辑
        :param batch: 但钱批次数据（输入序列，目标序列）
        :param batch_idx: 批次索引对象（is_train：是否是训练模式，is_last：是否是但钱epoch的最后一批）
        :return:
        """
        # 讲数据移动到GPU或者CPU中
        data,target=batch[0].to(self.device),batch[1].to(self.device)
        # 训练模式下的全局步数更新
        if self.mode.is_train:
            # 统计已处理的token总数，batch_size*seq_len
            tracker.add_global_step(data.shape[0]*data.shape[1])
        # 模型前向传播
        with  self.mode.update(is_log_activations=batch_idx.is_last):
            # 获取当前记忆
            mem=self.memory.get()
            # 讲记忆传入模型中推理返回新的记忆
            output,new_men=self.model(data,mem)
            # 合并压缩记忆
            mem,mem_to_compress=self.merge_compress_memory(mem,new_men)
            # 更新记忆状态
            self.memory.set(mem)
        # 交叉熵计算损失
        loss=self.loss_func(output,target)
        tracker.add('loss.',loss)
        # 如果记忆被压缩
        if mem_to_compress:
            ar_loss=self.attention_reconstruction_loss(new_men,mem_to_compress)

            tracker.add('ar_loss.',ar_loss)

            loss+=ar_loss  # 总损失=主损失+记忆重建损失

        self.accuracy(output, target)  # 计算当前批次的准确率
        self.accuracy.track() # 记录到track
        # 返现的给传播
        if self.mode.is_train:
            loss.backward()    # 反向传播
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            self.optimizer.step() # 参数更新
            if batch_idx.is_last:  # 如果是最后一批
                tracker.add('model',self.model)  # 记录模型状态

            # 梯度清零
            self.optimizer.zero_grad()
        # 保存结果将跟踪数据写入日志
        tracker.save()

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






















