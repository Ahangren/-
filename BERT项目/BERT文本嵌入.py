from typing import List

import torch
from transformers import BertTokenizer,BertModel

from labml import lab,monit

class BERTChunkEmbeddings:
    def __init__(self,device):
        self.device=device

        # 从HuggingFace加载BERT分词器
        with monit.section('Load BERT tokenizer'):
            self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased',
                                                         cache_dir=str(
                                                             lab.get_data_path()/'cache'/'bert-tokenizer'
                                                         ))
        # 从HuggingFace加载BERT模型
        with monit.section('Load BERT model'):
            self.model=BertModel.from_pretrained('bert-base-uncased',cache_dir=str(lab.get_data_path()/'cache'/'bert-model'))
            self.model.to(self.device)
    @staticmethod
    def _trim_chunk(chunk):
        striped=chunk.strip()
        parts=striped.split()
        striped=striped[len(parts[0]):len(parts[-1])]
        striped=striped.strip()

        if not striped:
            return chunk
        else:
            return striped
    def __call__(self, chunks):
        with torch.no_grad():
            # 处理每个句子的空格和第一个词和最后一个词
            trimmed_chunks=[self._trim_chunk(c) for c in chunks]
            # 将句子转换成模型能懂的序列，用tokenizer将句子变成数字ID，bert模型中游个字典，每个词都有一个对应的id
            tokens=self.tokenizer(trimmed_chunks,return_tensors='pt',add_special_tokens=False)
            input_ids = tokens['input_ids'].to(self.device)
            # 同时生成两个辅助标记。attention_mask和token_type_ids
            #attention_mask: 在bert分词中会有填充符和空白符，这个就是用来记录的，是词的地方为1是空白的位置为0，训练的时候关注1的位置就可以了
            attention_mask=tokens['attention_mask'].to(self.device)
            # 这个主要是用给双句子输出的，第一个句子的词的地方为0，第二个句子的词的地方为1
            token_type_ids=tokens['token_type_ids'].to(self.device)
            # 将这些数字给bert模型，模型会生成每个字的向量last_hidden_state[batch_size,序列长度,隐藏层长度]
            output=self.model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
            # 把所有的字的向量按重要性加权平均，用attention_mask当权重
            """
            输入文本: "你好"
       ↓
            Tokenizer转换:
               input_ids = [CLS]你 好[SEP] → [101, 1001, 1002, 102]
               attention_mask = [1, 1, 1, 1]
                   ↓
            BERT模型计算:
               每个ID通过12层Transformer
                   ↓
            输出: 
               - "你" → 768维向量 [0.3, 0.4, ...]
               - "好" → 768维向量 [0.5, 0.6, ...]
            """
            state=output['last_hidden_state']
            emb=(state*attention_mask[:,:,None]).sum(dim=1)/attention_mask[:,:,None].sum(dim=1)
            # 最终得到[0.25,0.3,......](768维的句子向量)
            """
            输入：["你好"] + 填充到长度3
            字向量： [你] [好] [PAD]
                     0.1  0.3   0.0
                     0.2  0.4   0.0
            mask：    1    1     0
            
            加权平均 = ( [0.1,0.2] + [0.3,0.4] + [0,0] ) / (1+1+0) = [0.2, 0.3]
            """
            # 过程：文本输入 -> 清理 -> 转token ID -> BERT处理 -> 字向量 -> 加权平均 -> 句子向量
            return emb

