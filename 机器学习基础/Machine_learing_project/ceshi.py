# 姓名：程烁
# 学号：2022b05030

import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文显示，防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置token
ts.set_token('cd138a8fe8b0831bc2f176a71c6f76313c21c65d076725ba559cdffc')
pro = ts.pro_api()

# 1. 数据获取部分修改为2024年1月1日至2025年3月1日
print('正在获取数据...')
df = pro.daily(ts_code='000045.SZ',
               start_date='20240101',
               end_date='20250301')

# 保存数据
df.to_csv('000045.csv',index=False)
print('数据shape:',df.shape)
print('\n前10条数据:')
print(df.head(10))

# 2. 绘制成交额直方图
plt.figure(figsize=(10,6))
plt.hist(df['amount'], bins=30)
plt.title('000045股票成交额')  # 设置标题
plt.xlabel('成交额(元)')
plt.ylabel('频次')
plt.savefig('amount_histogram_000045.png')
plt.close()

# 3. 数据降维部分修改为[收盘价、最高价、最低价]降成1维
print('\n开始PCA降维...')
df_pca = df[['close', 'high', 'low']]  # 按要求选择特征
scaler_pca = StandardScaler()
df_scaled = scaler_pca.fit_transform(df_pca)

pca = PCA(n_components=1)  # 降为1维
df_pca_result = pca.fit_transform(df_scaled)

# 绘制PCA降维结果曲线
plt.figure(figsize=(12, 6))
plt.plot(df_pca_result, label='PCA降维结果')
plt.title('PCA降维-程烁2022b05030')  # 按要求修改标题
plt.xlabel('时间（交易日）')
plt.ylabel('主成分值')
plt.legend()
plt.grid(True)
plt.savefig('pca_result.png')
plt.close()

# 4. 数据预测部分修改
# 使用前5+0=5个交易日数据预测（学号最后一位为0）
lookback = 5
features = ['open', 'high', 'low', 'close', 'vol', 'amount']

# 创建特征和目标
X = []
y = []
for i in range(len(df) - lookback):
    window = df[features].iloc[i:i+lookback].values
    target = df['close'].iloc[i+lookback]
    X.append(window)
    y.append(target)

X = np.array(X)
y = np.array(y)

# 数据标准化
scaler = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[2])
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(X.shape)

# PCA降维（对每日特征降维）
n_components = 1  # 降为1维
pca = PCA(n_components=n_components)

X_pca = []
for sample in X_scaled:
    sample_pca = pca.fit_transform(sample)
    X_pca.append(sample_pca)
X_pca = np.array(X_pca)

# 展平特征用于SVR
X_final = X_pca.reshape(X_pca.shape[0], -1)

# 修改测试集为最后30个交易日（按要求）
X_train, X_test = X_final[:-30], X_final[-30:]
y_train, y_test = y[:-30], y[-30:]

# 标准化目标变量
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

# 使用两种核函数的SVR（按要求）
print('\n开始SVR预测...')
# 第一种：rbf核函数
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X_train, y_train_scaled)
y_pred_rbf_scaled = svr_rbf.predict(X_test)
y_pred_rbf = y_scaler.inverse_transform(y_pred_rbf_scaled.reshape(-1, 1)).ravel()

# 第二种：linear核函数
svr_linear = SVR(kernel='linear', C=100)
svr_linear.fit(X_train, y_train_scaled)
y_pred_linear_scaled = svr_linear.predict(X_test)
y_pred_linear = y_scaler.inverse_transform(y_pred_linear_scaled.reshape(-1, 1)).ravel()

# 评估模型
mse_rbf = mean_squared_error(y_test, y_pred_rbf)
r2_rbf = r2_score(y_test, y_pred_rbf)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print('RBF核函数SVR结果:')
print(f'均方误差(MSE): {mse_rbf:.2f}')
print(f'R平方值: {r2_rbf:.2f}')

print('\nLinear核函数SVR结果:')
print(f'均方误差(MSE): {mse_linear:.2f}')
print(f'R平方值: {r2_linear:.2f}')

# 绘制两种核函数的预测结果（按要求分开绘制）
# 1. RBF核函数结果
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='真实数据', linewidth=2)
plt.plot(y_pred_rbf, label='RBF核预测', linestyle='--')
plt.title('深科技股份有限公司股票预测曲线分析图（RBF核）')  # 按要求修改标题
plt.xlabel('时间（交易日）')
plt.ylabel('收盘价')
plt.legend()
plt.grid(True)
plt.savefig('svr_rbf_prediction.png')
plt.close()

# 2. Linear核函数结果
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='真实数据', linewidth=2)
plt.plot(y_pred_linear, label='Linear核预测', linestyle='--')
plt.title('深科技股份有限公司股票预测曲线分析图（Linear核）')  # 按要求修改标题
plt.xlabel('时间（交易日）')
plt.ylabel('收盘价')
plt.legend()
plt.grid(True)
plt.savefig('svr_linear_prediction.png')
plt.close()

print('\n程序运行完成!')