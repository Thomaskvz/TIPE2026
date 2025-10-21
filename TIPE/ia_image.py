import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def charge_data(dossier):
    images = []
    for fichier in os.listdir(dossier):
        img_path = os.path.join(dossier, fichier)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img = img/255
        images.append(img.flatten())
    return images

data_avance = charge_data("training_data/avance")
data_droite = charge_data("training_data/droite")
data_gauche = charge_data("training_data/gauche")
data_recule = charge_data("training_data/recule")

print(len(data_avance))
print(len(data_avance) + len(data_droite) + len(data_gauche) + len(data_recule))

definition = ["Avance", "Droite", "Gauche", "Recule"]

X = np.vstack((data_avance,data_droite,data_gauche,data_recule))
y = np.array([0]*len(data_avance) + [1]*len(data_droite) + [2]*len(data_gauche) + [3]*len(data_recule))

print(len(X))
print(len(y))


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

def predict_image(img_path, model):
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img = img/255
    pred = model.predict(img.flatten().reshape(1,-1))
    return pred

test_prediction = predict_image("test/img_test.jpg", clf)
print(f"Prédiciton: {definition[test_prediction[0]]}")
print(f"Prédiciton: {test_prediction}")