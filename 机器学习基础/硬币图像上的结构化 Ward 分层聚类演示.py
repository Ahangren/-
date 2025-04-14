from skimage.data import coins  # skimage中的硬币图片
import numpy as np
import time
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from sklearn.cluster import AgglomerativeClustering

# 导入图片
orig_coins=coins()
# 对图片进行高斯模糊操作
smoothened_coins=gaussian_filter(orig_coins,sigma=2)
# 讲图片缩放到20%的大小
rescaled_coins=rescale(smoothened_coins,0.2,mode='reflect',anti_aliasing=False)

print(rescaled_coins)
# 讲形状改成一列多行的形式
X=np.reshape(rescaled_coins,(-1,1))
print(X)

# 定义数据结构
# 网格分组工具
from sklearn.feature_extraction.image import grid_to_graph
# 生成一个维度和rescaled_coins的网格
connectivity=grid_to_graph(*rescaled_coins.shape)


# 计算集群
print("计算结构化分层聚类")
st=time.time()
n_clusters=27
# 分组操作，颜色相似的像素要分到同一组，相邻的像素才能合并 (connectivity参数控制)，优先合并那些合并后"整体颜色差异最小"的区域
ward=AgglomerativeClustering(n_clusters=n_clusters,linkage='ward',connectivity=connectivity)
ward.fit(X)
label=np.reshape(ward.labels_,rescaled_coins.shape)
print(f"Elapsed time: {time.time() - st:.3f}s")
print(f"Number of pixels: {label.size}")
print(f"Number of clusters: {np.unique(label).size}")

# 在图像上分割结果

import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.imshow(rescaled_coins)
for i in range(n_clusters):
    plt.contour(
        label==i,
        colors=[plt.cm.nipy_spectral(i/float(n_clusters))]
    )
plt.axis('off')
plt.show()
