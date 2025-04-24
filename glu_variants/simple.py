import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import math


class AutoregressiveModule(nn.Module):
    def __init__(self, src_embed, encoder, generator):
        super().__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.generator = generator
        self.register_buffer("src_mask", None)

    def forward(self, src):
        # src shape: (batch_size, seq_len)
        if self.src_mask is None or self.src_mask.size(0) != src.size(1):
            # 生成正确的注意力掩码形状：(seq_len, seq_len)
            self.src_mask = subsequent_mask(src.size(1)).to(src.device)

        embedded = self.src_embed(src)  # (batch_size, seq_len, d_model)
        encoded = self.encoder(embedded, self.src_mask)  # (batch_size, seq_len, d_model)
        return self.generator(encoded)  # (batch_size, seq_len, vocab_size)


def subsequent_mask(size):
    """生成下三角注意力掩码"""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return ~mask  # True表示可以关注的位置


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, glu_variant='Bilinear'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # GLU变体实现
        if glu_variant == 'GLU':
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.Sigmoid(),
                nn.Linear(d_ff, d_model)
            )
        elif glu_variant == 'Bilinear':
            # 修正权重矩阵形状定义
            self.w1 = nn.Parameter(torch.Tensor(d_model, d_ff))  # (d_model, d_ff)
            self.v = nn.Parameter(torch.Tensor(d_model, d_ff))  # (d_model, d_ff)
            self.w2 = nn.Parameter(torch.Tensor(d_ff, d_model))  # (d_ff, d_model)

            nn.init.xavier_uniform_(self.w1)
            nn.init.xavier_uniform_(self.v)
            nn.init.xavier_uniform_(self.w2)

            # 修正lambda函数实现
            self.ffn = lambda x: torch.matmul(
                torch.matmul(x, self.w1) * torch.matmul(x, self.v),
                self.w2
            )

        elif glu_variant == 'ReGLU':
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            )
        elif glu_variant == 'GEGLU':
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            )
        elif glu_variant == 'SwiGLU':
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, d_model)
            )
        else:
            raise ValueError(f'不支持的GLU变体: {glu_variant}')

    def forward(self, src, src_mask=None):
        # 调整形状以适应MultiheadAttention
        src = src.transpose(0, 1)  # (seq_len, batch_size, d_model)

        # 处理注意力掩码
        attn_mask = None
        if src_mask is not None:
            if src_mask.dim() == 3:
                attn_mask = src_mask[0]  # 取第一个batch的掩码
            elif src_mask.dim() == 2:
                attn_mask = src_mask

        # 自注意力计算
        src2, _ = self.self_attn(
            src, src, src,
            attn_mask=attn_mask,
            key_padding_mask=None
        )

        # 恢复原始形状
        src = src.transpose(0, 1)  # (batch_size, seq_len, d_model)
        src2 = src2.transpose(0, 1)

        # 残差连接和归一化
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络部分
        original_shape = src.shape  # (batch_size, seq_len, d_model)
        src = src.reshape(-1, original_shape[-1])  # (batch_size*seq_len, d_model)

        # 使用修正后的矩阵乘法
        src2 = self.ffn(src)  # 现在形状正确

        # 恢复形状
        src2 = src2.reshape(original_shape)
        src = src.reshape(original_shape)

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout, glu_variant):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout, glu_variant)
            for _ in range(n_layers)
        ])

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class TinyShakespeareDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        return (
            self.data[idx:idx + self.seq_len],
            self.data[idx + 1:idx + self.seq_len + 1]
        )


def train_model():
    config = {
        'd_model': 256,
        'n_layers': 6,
        'n_heads': 8,
        'd_ff': 1024,
        'dropout': 0.1,
        'glu_variant': 'Bilinear',
        'batch_size': 32,  # 减小batch_size以适应内存
        'seq_len': 128,  # 减小序列长度
        'epochs': 10,  # 减少epochs用于测试
        'lr': 1.0,
        'warmup': 2000
    }

    # 数据加载
    with open('tinyshakespeare.txt', 'r') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    # 修复：正确使用seq_len和batch_size
    dataset = TinyShakespeareDataset(data, config['seq_len'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # 模型构建
    src_embed = nn.Sequential(
        nn.Embedding(vocab_size, config['d_model']),
        PositionalEncoding(config['d_model'], config['dropout'])
    )
    encoder = TransformerEncoder(
        config['n_layers'], config['d_model'], config['n_heads'],
        config['d_ff'], config['dropout'], config['glu_variant']
    )
    generator = nn.Linear(config['d_model'], vocab_size)
    model = AutoregressiveModule(src_embed, encoder, generator)

    # 模型的保存路径
    model_save_path='transformer_model.path'
    best_loss=float('inf')
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0

        for batch, (x, y) in enumerate(dataloader):
            # 学习率预热
            step_num = epoch * len(dataloader) + batch + 1
            lr = config['d_model']** (-0.5) * min(
                step_num **(-0.5),
            step_num * config['warmup'] ** (-1.5)
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # 前向传播
            logits = model(x)  # (batch_size, seq_len, vocab_size)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
        epoch_loss=total_loss/len(dataloader)
        print(f'Epoch {epoch}, Loss: {epoch_loss:.4f})')

        # 模型保存模块
        if epoch_loss<best_loss:
            best_loss=epoch_loss
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':epoch_loss,
                'config':config,
                'stoi':stoi,
                'itos':{i:ch for i ,ch in enumerate(chars)}
            },model_save_path)
            print(f'模型已经保存到{model_save_path}')
        # 文本生成
        if epoch % 2 == 0:  # 更频繁地生成样本
            model.eval()
            with torch.no_grad():
                start = "It is"
                x = torch.tensor([stoi[ch] for ch in start], dtype=torch.long).unsqueeze(0)
                # 自回归生成
                for _ in range(50):  # 生成50个字符
                    logits = model(x)
                    next_token = logits.argmax(-1)[:, -1]
                    x = torch.cat([x, next_token.unsqueeze(0)], dim=-1)
                generated = ''.join([chars[i] for i in x[0].tolist()])
                print(f'Generated: {generated}')
# 保存后的模型加载模块
def load_mode(model_save_path='transformer_model.path'):
    checkpoint=torch.load(model_save_path)

    # 重构模型
    config=checkpoint['config']
    chars=list(checkpoint['itos'].values())
    stoi=checkpoint['stoi']
    vocab_size=len(chars)

    # 加载模型参数
    src_embed=nn.Sequential(
        nn.Embedding(vocab_size,config['d_model']),
        PositionalEncoding(config['d_model'],config['dropout'])
    )
    encoder = TransformerEncoder(
        config['n_layers'], config['d_model'], config['n_heads'],
        config['d_ff'], config['dropout'], config['glu_variant']
    )
    generator=nn.Linear(config['d_model'],vocab_size)
    model=AutoregressiveModule(src_embed, encoder, generator)
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])

    return model,config,chars,stoi


def generate_text(model, stoi, itos, config, prompt="It is", max_length=100):
    model.eval()
    with torch.no_grad():
        # 初始化输入
        input_seq = torch.tensor([stoi[ch] for ch in prompt], dtype=torch.long).unsqueeze(0)
        generated = prompt

        for _ in range(max_length):
            # 生成下一个token
            logits = model(input_seq)
            next_token = logits.argmax(-1)[:, -1]

            # 添加到生成文本中
            generated += itos[next_token.item()]

            # 更新输入序列
            input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=-1)

            # 如果序列太长，只保留最后seq_len个token
            if input_seq.size(1) > config['seq_len']:
                input_seq = input_seq[:, -config['seq_len']:]

        return generated


if __name__ == '__main__':
    # 训练模型
    train_model()

    # 加载模型并生成文本
    model, config, chars, stoi = load_mode()
    itos = {i: ch for i, ch in enumerate(chars)}

    # 生成文本示例
    generated_text = generate_text(model, stoi, itos, config, prompt="It is")
    print("生成的文本:", generated_text)