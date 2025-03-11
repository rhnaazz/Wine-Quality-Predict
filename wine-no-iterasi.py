import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier

data = pd.read_csv("winequality.csv", sep=';')
print(data.columns)

# Preprocessing

data.isnull().sum()
data.dtypes
data['quality'].unique()
y = data['quality']
X = data.drop('quality', axis=1)

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(X)

# Clustering

kmeans = KMeans(n_clusters=6)
kmeans.fit(X)
clusters = kmeans.predict(X)
print(clusters)

# Visualization

pca = PCA(n_components=2)

reduced_X = pd.DataFrame(pca.fit_transform(X), columns=["PC1", "PC2"])
print(reduced_X)

reduced_X['cluster'] = clusters
print(reduced_X)

reduced_centers = pca.transform(kmeans.cluster_centers_)
# print(reduced_centers)

plt.figure(figsize=(14, 10))

colors = list(mcolors.TABLEAU_COLORS.values())
num_clusters = reduced_X['cluster'].nunique()

# Gunakan colormap jika jumlah cluster lebih banyak dari jumlah warna default
if num_clusters > len(colors):
    cmap = get_cmap("tab10", num_clusters)
    colors = [cmap(i) for i in range(num_clusters)]

# Plot setiap cluster dengan ukuran titik lebih kecil (s=10)
for cluster, color in zip(sorted(reduced_X['cluster'].unique()), colors):
    cluster_data = reduced_X[reduced_X['cluster'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], color=color, label=f'Cluster {cluster}', s=10)

# Plot centroid dengan ukuran yang lebih besar agar tetap terlihat
plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], color='black', marker='x', s=100, label="Centroids")

plt.legend()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Cluster Visualization")

plt.show()

# Training

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
model = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=500)

model.fit(X_train, y_train)
print(f"Model Accuracy: {model.score(X_test, y_test)}")
