import numpy as np
from sklearn.datasets import load_digits
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. 加载数据集
data, labels = load_digits(return_X_y=True)
(n_sample, n_features), n_digits = data.shape, np.unique(labels).size
print(f"总共有{n_digits}个类别，总共有{n_sample}个例子，总共有{n_features}个特征")


def bench_k_means(kmeans, name, data, labels):
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    # 修正这里：应该传入labels和estimator[-1].labels_，而不是inertia_
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    results += [
        metrics.silhouette_score(
            data, estimator[-1].labels_,
            metric='euclidean',
            sample_size=300,
        )
    ]

    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))


print(82 * "_")
print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name='k-means++', data=data, labels=labels)

kmeans = KMeans(init='random', n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name='random', data=data, labels=labels)

pca = PCA(n_components=n_digits).fit(data)
kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
bench_k_means(kmeans=kmeans, name='PCA-based', data=data, labels=labels)

print('-' * 82)

# 可视化部分
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=4)
kmeans.fit(reduced_data)

# 绘制决策边界
h = 0.02  # 网格步长
x_min,x_max=reduced_data[:,0].min()-1,reduced_data[:,0].max()+1
y_min,y_max=reduced_data[:,1].min()-1,reduced_data[:.1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max),np.arange(y_min,y_max))

# 预测每个网格点的标签
z=kmeans.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(z,interpolation='nearest',
           extent=(xx.min(),xx.max(),yy.min(),yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto',
           origin='lower')


plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# 绘制聚类中心
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()