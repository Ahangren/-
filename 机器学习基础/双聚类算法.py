from matplotlib import pyplot as plt
from sklearn.datasets import make_checkerboard
n_clusters=(4,3)
#make_checkerboard 是 scikit-learn 提供的一个用于生成双聚类测试数据的实用函数，它创建具有棋盘结构的合成数据集，特别适合用于测试和演示双聚类算法。
data,rows,columns=make_checkerboard(
    shape=(300,300),n_clusters=n_clusters,noise=10,shuffle=False,random_state=42
)
plt.matshow(data)
plt.title("原始数据")
plt.show()


# 上面得到的是有序的行和列，这里获得打乱的
import numpy as np
rng=np.random.RandomState()
row_idx_shuffle=rng.permutation(data.shape[0])
clu_idx_shuffle=rng.permutation(data.shape[1])

# 绘制打乱后的矩阵
print(data.shape)
data=data[row_idx_shuffle][:,clu_idx_shuffle]
print(data.shape)
plt.matshow(data)
plt.title('打乱以后的矩阵')
plt.show()


from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score
model=SpectralBiclustering(n_clusters=n_clusters,method='log',random_state=0)
model.fit(data)

score=consensus_score(
    model.biclusters_,(rows[:,row_idx_shuffle],columns[:,clu_idx_shuffle])
)
print(f'consensus score:{score:.1f}')

plt.matshow(
    np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1),
    cmap=plt.cm.Blues,
)
plt.title("Checkerboard structure of rearranged data")
plt.show()






