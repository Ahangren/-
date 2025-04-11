import torch
from torch.utils.data import DataLoader

from labml import experiment, tracker
from labml.configs import option, calculate
from labml_helpers.datasets.text import SequentialUnBatchedDataset
from labml_nn.transformers.alibi import AlibiMultiHeadAttention
from labml_nn.experiments.nlp_autoregression import transpose_batch
from labml_nn.transformers import TransformerConfigs
from labml_nn.transformers.gpt import Configs as GPTConfigs


class Configs(GPTConfigs):
    transformer : TransformerConfigs='GPT_ALiBi'

    valid_seq_len: int=128
    valid_loader='shuffled_longer_valid_loader'

    def other_metrics(self, output: torch.Tensor, target: torch.Tensor):
        if self.seq_len<output.shape[0]:
            tracker.add(f'loss.{self.seq_len - 1}.', self.loss_func(output[self.seq_len - 1], target[self.seq_len - 1]))
            tracker.add(f'loss.0.', self.loss_func(output[0], target[0]))
        tracker.add(f'loss.{int(output.shape[0]) - 1}.', self.loss_func(output[-1], target[-1]))
def _alibi_mha(c: TransformerConfigs):
    return AlibiMultiHeadAttention(c.n_heads, c.d_model, dropout_prob=c.dropout)

calculate(TransformerConfigs.encoder_attn, 'alibi_mha', _alibi_mha)
calculate(TransformerConfigs.decoder_attn, 'alibi_mha', _alibi_mha)
calculate(TransformerConfigs.decoder_mem_attn, 'alibi_mha', _alibi_mha)

@option(Configs.valid_loader)
def shuffled_longer_valid_loader(c: Configs):

    return DataLoader(SequentialUnBatchedDataset(text=c.text.valid,
                                                 dataset=c.text,
                                                 seq_len=c.valid_seq_len),
                      batch_size=c.batch_size,
                      collate_fn=transpose_batch,
                      shuffle=True)
@option(Configs.transformer, 'GPT_ALiBi')
def _transformer_configs(c: Configs):
    conf = TransformerConfigs()
    conf.n_src_vocab = c.n_tokens
    conf.n_tgt_vocab = c.n_tokens
    conf.ffn.activation = 'GELU'
    conf.src_embed = 'no_pos'

    conf.tgt_embed = 'no_pos'

    conf.encoder_attn = 'alibi_mha'

    conf.decoder_attn = 'alibi_mha'

    conf.decoder_mem_attn = 'alibi_mha'
    return conf

def main():
    experiment.create(name="gpt_alibi")
    conf = Configs()
    experiment.configs(conf, {
        'tokenizer': 'character',
        'prompt_separator': '',
        'prompt': 'It is ',
        'text': 'tiny_shakespeare',
        'seq_len': 64,
        'valid_seq_len': 80,
        'epochs': 128,
        'batch_size': 128,
        'transformer.d_model': 128,
        'transformer.ffn.d_ff': 512,

    'transformer.n_heads': 8,

    'transformer.n_layers': 4,

    'transformer.dropout': 0.1,
 })
    experiment.add_pytorch_models({'model': conf.model})
    with experiment.start():
        conf.run()
if __name__ == '__main__':
   main()