## PARTIE 01
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

## import
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import joblib


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

## PARTIE 03

X_entrainement, X_autre, y_entrainement, y_autre = train_test_split(X, y, test_size=0.3)  # 70% d'entrainement et 30% de test et de validation
X_validation, X_test, y_validation, y_test = train_test_split(X_autre, y_autre, test_size=1/3)  # 10% de validation et  20% de test

modele = RandomForestClassifier()

modele.fit(X_entrainement, y_entrainement)

prediction = modele.predict(X_test)

precision = accuracy_score(y_test, prediction)

precision_recall_f1Score = classification_report(y_test, prediction)

matrice_confusion = confusion_matrix(y_test, prediction)

print("La performance du modèle Random Forest sur l'ensemble de test généré est :")
print(f"La précision du test est: {precision:.4f}")
print("La précision - Recall - Score F1:")
print(precision_recall_f1Score)
print("La matrice de Confusion:")
print(matrice_confusion)

###Partie 4

df_X_validation = pd.DataFrame(X_validation)
df_y_validation = pd.DataFrame(y_validation, columns=['target'])

df_X_validation.to_csv('X_validation.csv', index=False)
df_y_validation.to_csv('y_validation.csv', index=False)

joblib.dump(modele, 'modele_entraine.joblib')
