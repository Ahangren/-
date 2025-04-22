import torch
from networkx import neighbors
from torch import nn
from torch.utils.data import DataLoader,RandomSampler

from labml import monit,lab,tracker,experiment,logger
from labml.logger import Text
from labml_helpers.datasets.text import TextFileDataset
from labml_nn.optimizers.noam import Noam
from labml_nn.transformers.retro import model as retro
from labml_nn.transformers.retro.dataset import Dataset,RetroIndex
from labml_nn.transformers.retro.model import RetroModel,NearestNeighborEncoder


class Sample:
    def __init__(self,device,model,tds,chunk_len):
        self.chunk_len=chunk_len
        self.tds=tds
        self.model=model
        self.device=device

        self.index=RetroIndex()

    def retrieve_nearest_neighbours(self,chunk):
        neighbor_offsets=self.index([chunk],None)
        text=self.tds.train
        neighbors=[text[j:j+self.chunk_len*2] for j in neighbor_offsets[0]]

        return neighbors

    def sample(self, prompt: str, sample_len: int):
        neighbors_str=[]
        sampled = ''
        # 基于prompt生成文本
        for i in range(sample_len):
            while len(neighbors_str) < len(prompt) // self.chunk_len:
                off = len(neighbors_str) * self.chunk_len
                chunk = prompt[off: off + self.chunk_len]
                neighbors_str.append(self.retrieve_nearest_neighbours(chunk))
            src = self.tds.text_to_i(prompt)
            neighbors = torch.stack([torch.stack([self.tds.text_to_i(n) for n in chunk]) for chunk in neighbors_str])
            src = src.to(self.device)

            neighbors = neighbors.to(self.device)
            res = self.model(src[None, :], neighbors[None, :, :, :])
            token = res[0, -1, :].argmax(dim=-1)

            prompt += self.tds.itos[token.item()]

            sampled += self.tds.itos[token.item()]
            return sampled

class Trainer:
    def __init__(self, device: torch.device, model: retro.RetroModel,
                                 dataloader: DataLoader, optimizer: torch.optim.Optimizer):



        self.optimizer = optimizer

        self.device = device

        self.dataloader = dataloader

        self.model = model

        self.loss_func = nn.CrossEntropyLoss()

    def __call__(self, *args, **kwargs):
        for i, (src, tgt, neighbors) in monit.enum('Train', self.dataloader):

            src, tgt, neighbors = src.to(self.device), tgt.to(self.device), neighbors.to(self.device)

            res = self.model(src, neighbors)

            loss = self.loss_func(res.view(-1, res.shape[-1]), tgt.view(-1))

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            tracker.save({'loss.train': loss})

            tracker.add_global_step(len(src))


def train():

    experiment.create(name='retro_small')
    device = torch.device('cuda:0')

    tds = TextFileDataset(

        lab.get_data_path() / 'tiny_shakespeare.txt',

        list,

        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

    train_dataset = Dataset(lab.get_data_path() / 'retro_train_dataset.json', tds)

    train_dl = DataLoader(train_dataset,

        batch_size = 4,
        sampler = RandomSampler(train_dataset, replacement=True))

    chunk_len = 16
    d_model = 128
    d_ff = 512
    n_heads = 16
    d_k = 16

    nearest_neighbor_encoder = NearestNeighborEncoder(chunk_len, 6, {3}, d_model, n_heads, d_k, d_ff)

    model = RetroModel(tds.n_tokens, d_model, 6,

        {3, 5},

        chunk_len, n_heads, d_k, d_ff,

        encoder = nearest_neighbor_encoder)

    model = model.to(device)

    optimizer = Noam(model.parameters(), lr=1., d_model=d_model, warmup=2_000)

    trainer = Trainer(device, model, train_dl, optimizer)

    sampler = Sample(device, model, tds, chunk_len)

    prompt = '''Second Citizen:\nOne word, good citizens.\n\nFirst Citizen:'''

    experiment.add_pytorch_models(model=model)

    with experiment.start():

        for epoch in monit.loop(32):

            trainer()

            tracker.new_line()

            logger.log([(prompt.replace('\n', '\\n\n'), Text.subtle),
                        (sampler.sample(prompt, 128).replace('\n', '\\n\n'), Text.none)])

            experiment.save_checkpoint()

if __name__ == '__main__':

    train()
