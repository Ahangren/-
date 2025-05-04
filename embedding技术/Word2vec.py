# 导入必要的库
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库
import torch.nn as nn  # PyTorch的神经网络模块
import torch.optim as optim  # PyTorch的优化器模块
import matplotlib.pyplot as plt  # 绘图库（虽然代码中未实际使用）
import torch.utils.data as Data  # PyTorch数据工具


# 定义数据类型和设备配置
dtype = torch.FloatTensor  # 默认张量类型（32位浮点数）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检测并选择GPU/CPU设备

# 超参数设置
batch_size = 8  # 每批次的样本数量
embedding_size = 2  # 词嵌入向量的维度
c = 2  # 上下文窗口大小（中心词前后各取2个词）


def prepare_data():
    # 定义训练用的示例句子
    sentences = ["longge like dog", "longge like cat", "longge like animal",
                 "dog cat animal", "banana apple cat dog like", "dog fish milklike",
                 "dog cat animal like", "longge like apple", "apple like", "longgelike banana",
                 "apple banana longge movie book music like", "cat dog hate", "catdog like"]

    # 将所有句子合并并分割成单词列表
    word_sequence = " ".join(sentences).split()

    # 构建词汇表（去除重复词）
    vocab = list(set(word_sequence))

    # 创建单词到索引的映射字典
    word2idx = {w: i for i, w in enumerate(vocab)}

    # 获取词汇表大小
    voc_size = len(vocab)

    # 用于存储skip-gram样本
    skip_grams = []
    print(word2idx)  # 打印词汇表映射关系

    # 生成skip-gram训练样本
    for idx in range(c, len(word_sequence) - c):
        center = word2idx[word_sequence[idx]]  # 获取中心词的索引
        # 获取上下文词的索引范围（前后各c个词）
        context_idx = list(range(idx - c, idx)) + list(range(idx + 1, idx + c + 1))
        # 将上下文词转换为索引
        context = [word2idx[word_sequence[i]] for i in context_idx]
        # 将每个(中心词,上下文词)对加入列表
        for w in context:
            skip_grams.append([center, w])

    # 定义数据生成函数
    def make_data(skip_grams):
        input_data = []
        output_data = []
        for i in range(len(skip_grams)):
            # 输入使用one-hot编码的中心词
            input_data.append(np.eye(voc_size)[skip_grams[i][0]])
            # 输出是上下文词的索引
            output_data.append(skip_grams[i][1])
        return input_data, output_data

    # 打印生成的skip-gram样本
    print(skip_grams)

    # 生成训练数据
    input_data, output_data = make_data(skip_grams)
    print(input_data)  # 打印输入数据（one-hot编码）
    print(output_data)  # 打印输出数据（目标词索引）

    # 将数据转换为PyTorch张量
    input_data, output_data = torch.Tensor(input_data), torch.LongTensor(output_data)

    # 创建数据集和数据加载器
    dataset = Data.TensorDataset(input_data, output_data)
    loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # 返回词汇表、词汇表大小和数据加载器
    return vocab, voc_size, loader


# 构建模型
# 定义Word2Vec模型类
class word2vec(nn.Module):
    def __init__(self, voc_size, dtype):
        super(word2vec, self).__init__()  # 调用父类初始化
        # 定义输入权重矩阵（从词汇表到嵌入层）
        self.w_in = nn.Parameter(torch.randn(voc_size, embedding_size).type(dtype))
        # 定义输出权重矩阵（从嵌入层回到词汇表）
        self.w_out = nn.Parameter(torch.randn(embedding_size, voc_size).type(dtype))

    def forward(self, x):
        # 前向传播过程
        hidden_layer = torch.matmul(x, self.w_in)  # 输入层到隐藏层的矩阵乘法
        output_layer = torch.matmul(hidden_layer, self.w_out)  # 隐藏层到输出层的矩阵乘法
        return output_layer  # 返回预测结果


# 定义训练函数
def train_model(voc_size, dtype, loader, vocab):
    # 初始化模型并移动到指定设备
    model = word2vec(voc_size, dtype).to(device)
    # 定义损失函数（交叉熵损失）
    criterion = nn.CrossEntropyLoss().to(device)
    # 定义优化器（Adam）
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环（20个epoch）
    for epoch in range(20):
        # 遍历数据加载器中的每个批次
        for i, (batch_x, batch_y) in enumerate(loader):
            # 将数据移动到指定设备
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 前向传播
            output = model(batch_x)
            # 计算损失
            loss = criterion(output, batch_y)

            # 反向传播
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

    # 可视化词向量
    for i, label in enumerate(vocab):
        # 获取模型参数
        w, wt = model.parameters()
        # 获取当前单词的嵌入向量坐标
        x, y = float(w[i][0]), float(w[i][1])
        # 绘制散点图
        plt.scatter(x, y, label=label)
        # 添加标签注释
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')
    # 显示图形
    plt.show()


# 准备数据
vocab, voc_size, loader = prepare_data()

# 主程序入口
if __name__ == '__main__':
    # 训练模型
    train_model(voc_size, dtype, loader, vocab)
