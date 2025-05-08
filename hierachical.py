
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage

iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target (flower species)

Z = linkage(X, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Index of Samples")
plt.ylabel("Euclidean Distance")
plt.show()

agg_clust = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_agg = agg_clust.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_agg, cmap='rainbow')
plt.title('Hierarchical Clustering of Iris Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

