# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 进度条显示

# ==================== 配置参数 ====================
IMAGE_SIZE = (128, 128)  # 图像输入尺寸
BATCH_SIZE = 32  # 批量大小（根据GPU内存调整）
EPOCHS = 20  # 训练轮次
LEARNING_RATE = 0.001  # 初始学习率
PATIENCE = 5  # 早停耐心值
NUM_WORKERS = 4  # 数据加载的线程数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择设备

# ==================== 数据准备 ====================
# 1. 加载文件名和标签
filenames = os.listdir("./data/kaggle数据集/train")
# 使用列表推导式快速生成标签（狗为1，猫为0）
categories = [1 if fname.split('.')[0] == 'dog' else 0 for fname in filenames]

# 2. 创建DataFrame并划分训练集/验证集
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
# 使用分层分割保持类别比例
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['category']  # 保持类别分布
)

# ==================== 数据增强 ====================
# 训练集的数据增强（防止过拟合）
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),  # 调整尺寸
    transforms.RandomRotation(15),  # 随机旋转(-15,15)度
    transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
    transforms.ColorJitter(  # 颜色抖动
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),  # 转为Tensor并归一化到[0,1]
    transforms.Normalize(  # 标准化（ImageNet均值/标准差）
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 验证集的数据处理（不做增强，只做标准化）
val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==================== 自定义数据集类 ====================
class DogCatDataset(Dataset):
    """自定义数据集类，继承自torch.utils.data.Dataset"""

    def __init__(self, df, root_dir, transform=None):
        """
        参数:
            df: 包含文件名和标签的DataFrame
            root_dir: 图像根目录
            transform: 要应用的数据增强
        """
        self.df = df.reset_index(drop=True)  # 重置索引确保连续性
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """返回数据集大小"""
        return len(self.df)

    def __getitem__(self, idx):
        """
        获取单个样本
        返回:
            image: 处理后的图像张量
            label: 对应的标签
        """
        # 1. 从DataFrame中获取文件名和标签
        img_name = self.df.loc[idx, 'filename']
        label = self.df.loc[idx, 'category']

        # 2. 构建完整图像路径并加载
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # 确保RGB格式

        # 3. 应用数据增强（如果有）
        if self.transform:
            image = self.transform(image)

        return image, label


# ==================== 创建数据加载器 ====================
# 训练集
train_dataset = DogCatDataset(
    df=train_df,
    root_dir="./data/kaggle数据集/train",
    transform=train_transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 训练时打乱数据
    num_workers=NUM_WORKERS,  # 多线程加载
    pin_memory=True  # 加速数据传到GPU
)

# 验证集
val_dataset = DogCatDataset(
    df=val_df,
    root_dir="./dogs-vs-cats/train",
    transform=val_transform
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # 验证时不打乱
    num_workers=NUM_WORKERS
)


# ==================== 模型定义 ====================
class DogCatClassifier(nn.Module):
    """自定义CNN模型"""

    def __init__(self):
        super().__init__()

        # 特征提取器（卷积层）
        self.features = nn.Sequential(
            # 第一卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # 第二卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # 第三卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # 分类器（全连接层）
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平卷积输出

            # 第一全连接层
            nn.Linear(128 * 16 * 16, 512),  # 输入尺寸计算：128x128经过3次池化→16x16
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # 输出层（二分类）
            nn.Linear(512, 2)
        )

    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==================== 初始化模型 ====================
model = DogCatClassifier().to(DEVICE)  # 将模型移到指定设备

# 使用预训练权重（可选）
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 2)  # 修改最后一层
# model = model.to(DEVICE)

# ==================== 训练配置 ====================
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（包含Softmax）
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)  # 使用AdamW优化器

# 学习率调度器（验证精度不提升时降低学习率）
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',  # 监控验证精度
    patience=2,  # 2个epoch无改善则调整
    factor=0.5,  # 学习率乘以0.5
    verbose=True  # 打印调整信息
)

# ==================== 训练循环 ====================
best_val_acc = 0.0  # 记录最佳验证精度
best_model_weights = None  # 保存最佳模型权重

# 训练日志
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

for epoch in range(EPOCHS):
    # === 训练阶段 ===
    model.train()  # 设置训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用tqdm显示进度条
    train_loop = tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{EPOCHS}')
    for images, labels in train_loop:
        # 1. 数据转移到设备
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # 2. 前向传播
        optimizer.zero_grad()  # 清零梯度
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 3. 反向传播和优化
        loss.backward()
        optimizer.step()

        # 4. 统计指标
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条信息
        train_loop.set_postfix({
            'loss': running_loss / (total / BATCH_SIZE),
            'acc': 100 * correct / total
        })

    # 计算本轮平均训练指标
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)

    # === 验证阶段 ===
    model.eval()  # 设置评估模式
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        val_loop = tqdm(val_loader, desc=f'Val Epoch {epoch + 1}/{EPOCHS}')
        for images, labels in val_loop:
            # 1. 数据转移到设备
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # 2. 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 3. 统计指标
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条信息
            val_loop.set_postfix({
                'loss': val_loss / (total / BATCH_SIZE),
                'acc': 100 * correct / total
            })

    # 计算本轮验证指标
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # 打印本轮摘要
    print(f'\nEpoch {epoch + 1} Summary:')
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

    # === 模型保存与早停 ===
    # 1. 更新学习率（基于验证精度）
    scheduler.step(val_acc)

    # 2. 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_weights = model.state_dict().copy()  # 深拷贝模型权重
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model_weights,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': best_val_acc,
        }, 'best_model.pth')
        print(f'>>> Best model saved with val_acc = {best_val_acc:.2f}%')

    # 3. 早停检查
    if (epoch - history['val_acc'].index(max(history['val_acc']))) >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch + 1}!')
        break

# 训练结束后加载最佳模型
model.load_state_dict(best_model_weights)
print(f'\nTraining completed. Best val_acc = {best_val_acc:.2f}%')

# ==================== 测试与可视化 ====================
# 1. 加载测试数据
test_df = pd.DataFrame({
    'filename': os.listdir("./data/kaggle数据集/test1")
})

# 2. 创建测试数据加载器
test_transform = val_transform  # 使用与验证集相同的变换
test_dataset = DogCatDataset(
    df=test_df,
    root_dir="./data/kaggle数据集/test1",
    transform=test_transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# 3. 预测测试集
model.eval()
predictions = []
with torch.no_grad():
    for images, _ in tqdm(test_loader, desc='Predicting'):
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

# 4. 保存预测结果
test_df['category'] = predictions
test_df['category'] = test_df['category'].map({0: 'cat', 1: 'dog'})
test_df.to_csv('./data/kaggle数据集/predictions.csv', index=False)

# 5. 可视化样本结果
sample = test_df.sample(18, random_state=42)  # 随机选取18个样本
plt.figure(figsize=(12, 12))
for idx, (_, row) in enumerate(sample.iterrows()):
    # 加载原始图像（未经标准化）
    img = Image.open(f"./data/kaggle数据集/test1/{row['filename']}").resize((128, 128))

    # 创建子图
    plt.subplot(6, 3, idx + 1)
    plt.imshow(img)
    plt.title(f"{row['filename']}\n({row['category']})", fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.savefig('./data/kaggle数据集/sample_predictions.png', dpi=300)
plt.show()

# ==================== GUI界面 ====================
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk


class DogCatApp(tk.Tk):
    """猫狗分类GUI应用"""

    def __init__(self):
        super().__init__()

        # 窗口设置
        self.title("Cat vs Dog Classifier")
        self.geometry("800x600")
        self.configure(bg='#f0f0f0')

        # 加载模型
        self.model = DogCatClassifier().to(DEVICE)
        self.model.load_state_dict(torch.load('./data/kaggle数据集/best_model.pth'))
        self.model.eval()

        # 创建UI组件
        self.create_widgets()

    def create_widgets(self):
        """创建所有GUI组件"""

        # 标题
        self.title_label = tk.Label(
            self,
            text="Cat vs Dog Classifier",
            font=('Helvetica', 20, 'bold'),
            bg='#f0f0f0',
            fg='#333333'
        )
        self.title_label.pack(pady=20)

        # 图像显示区域
        self.image_frame = tk.Frame(self, bg='#ffffff', bd=2, relief=tk.SUNKEN)
        self.image_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.image_frame, bg='#ffffff')
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 结果标签
        self.result_label = tk.Label(
            self,
            text="Upload an image to classify",
            font=('Helvetica', 14),
            bg='#f0f0f0',
            fg='#555555'
        )
        self.result_label.pack(pady=10)

        # 上传按钮
        self.upload_btn = ttk.Button(
            self,
            text="Upload Image",
            command=self.upload_image,
            style='Accent.TButton'
        )
        self.upload_btn.pack(pady=20)

        # 进度条
        self.progress = ttk.Progressbar(
            self,
            orient=tk.HORIZONTAL,
            length=300,
            mode='determinate'
        )
        self.progress.pack(pady=10)
        self.progress.pack_forget()  # 初始隐藏

        # 样式配置
        self.style = ttk.Style()
        self.style.configure('Accent.TButton', font=('Helvetica', 12), foreground='white')

    def upload_image(self):
        """处理图像上传"""
        try:
            # 打开文件对话框
            file_path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png")]
            )

            if not file_path:  # 用户取消选择
                return

            # 显示进度条
            self.progress.pack(pady=10)
            self.progress['value'] = 20
            self.update()

            # 加载并显示图像
            img = Image.open(file_path)
            img.thumbnail((400, 400))  # 缩略图

            # 更新进度
            self.progress['value'] = 50
            self.update()

            # 显示图像
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            # 更新进度
            self.progress['value'] = 70
            self.update()

            # 预处理图像并预测
            transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            # 预测
            with torch.no_grad():
                output = self.model(img_tensor)
                _, pred = torch.max(output, 1)

            # 更新结果
            result = "It's a Dog!" if pred.item() == 1 else "It's a Cat!"
            self.result_label.config(
                text=result,
                fg='#0066cc' if pred.item() == 1 else '#cc3300'
            )

            # 完成进度条
            self.progress['value'] = 100
            self.update()
            self.after(1000, self.progress.pack_forget)  # 1秒后隐藏

        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}", fg='red')
            self.progress.pack_forget()


# 运行GUI应用
if __name__ == "__main__":
    app = DogCatApp()
    app.mainloop()