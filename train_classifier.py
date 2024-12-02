X, y = make_classification(
    n_samples=10000,
    n_informative=10,
    n_classes=3, 
    n_clusters_per_class=2,
    class_sep=2.0
)

## PARTIE 02
""" import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# on utilise la méthode de réduction de dimension (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# on visualise des données en 2D
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.7)
plt.colorbar(scatter)
plt.title("Données (PCA)")
plt.show() """
