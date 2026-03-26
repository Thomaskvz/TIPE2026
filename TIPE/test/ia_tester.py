import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt


def charge_data(dossier):
    images = []
    for fichier in os.listdir(dossier):
        img_path = os.path.join(dossier, fichier)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (80, 30))  # Redimensionner les images à une taille fixe
        img = img/255
        images.append(img.flatten())
    return images

definition = ["Avance", "Droite", "Gauche"]

knn = 9

print("Chargement des données...")
data_avance = charge_data("training_data/avance")
data_droite = charge_data("training_data/droite")
data_gauche = charge_data("training_data/gauche")
data_recule = charge_data("training_data/recule")

cv2.imwrite("test/test_avance.png", data_avance[0].reshape(30, 80)*255)

print(f"Nombre d'images avance: {len(data_avance)}")
print(f"Nombre d'images total: {len(data_avance) + len(data_droite) + len(data_gauche) + len(data_recule)}")

print("Compilation des images...")
X = np.vstack((data_avance,data_droite,data_gauche,data_recule))
print("Attribution des valeurs...")
y = np.array([0]*len(data_avance) + [1]*len(data_droite) + [2]*len(data_gauche) + [3]*len(data_recule))

print("Séparation en images d'entrainement et de test...")
# 20% des images sont utilisées en tant que test

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

print("Entraînement du modèle...")

# clf = SVC(kernel='linear')
clf = MLPClassifier(
    hidden_layer_sizes=(32,),  # one hidden layer
    activation='relu',
    solver='adam',
    max_iter=300
)
# clf = KNeighborsClassifier(n_neighbors=knn)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f"Précision: {accuracy_score(y_test, y_pred)}")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=definition)
disp.plot() #cmap=plt.cm.Blues
plt.title(f"Confusion Matrix (k={knn})")
plt.show()
