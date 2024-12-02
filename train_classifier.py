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
