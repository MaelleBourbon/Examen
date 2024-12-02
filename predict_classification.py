import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# on importe le modèle entraîné
modele = joblib.load('modele_entraine.joblib')  # Charger le modèle depuis le fichier .joblib

# on charge les données de validation
X_validation = pd.read_csv('X_validation.csv')  # Charger X_validation depuis le fichier CSV
y_validation = pd.read_csv('y_validation.csv')  # Charger y_validation depuis le fichier CSV

X_validation1 = X_validation.values  
y_validation1 = y_validation['target'].values 

# on prédit sur les données de validation 
predictions = modele.predict(X_validation1)

# on évalue la performance

# calcul de l'accurance
precision = accuracy_score(y_validation1, predictions)

# calcul de la precision, le recall et le F1 Score
precision_recall_f1Score = classification_report(y_validation1, predictions)

# calcul de la matrice de confusion
matrice_confusion = confusion_matrix(y_validation1, predictions)

print("La performance du modèle Random Forest sur l'ensemble de validation généré est :")
print(f"La précision de l'ensemble validation est: {precision:.4f}")
print("La précision - Recall - Score F1:")
print(precision_recall_f1Score)
print("La matrice de Confusion:")
print(matrice_confusion)

