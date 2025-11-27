def predDet(img,hauteur):
    i = 0
    while i < len(img[0]) and img[hauteur,i] < 0.8:
        i += 1
    if i == len(img[0]):        # Pas de blanc: Recule
        return 3
    if i >= len(img[0])//2:     # Blanc à droite: Droite
        return 2
    
    j = len(img[0]) -1
    while j < len(img[0]) and img[hauteur,j] < 0.8:
        j -= 1

    if j < len(img[0])//2:     # Blanc à gauche: Gauche
        return 1
    
    else:                      # Blanc aux 2: Avance
        return 0
    

    # if img[240, 50] > 0.7 and img[240, 540] > 0.7:
    #     return 0
    # if img[240, 50] > 0.7:
    #     return 1
    # if img[240, 540] > 0.7:
    #     return 2
    # else:
    #     return 3
    
if __name__ == "__main__":
    import cv2
    import numpy as np

    dossier = "training_data/gauche"
    nom = "156"
    test_img = cv2.imread(f"{dossier}/{nom}.jpg", cv2.IMREAD_GRAYSCALE)
    test_img = test_img / 255


    print(test_img[240, 50], test_img[240, 540])
    prediction = predDet(test_img, 200)
    definition = ["Avance", "Droite", "Gauche", "Recule"]
    print(f"Prédiction déterministe: {definition[prediction]}")



    # while True:
    #     cv2.imshow("Test Image", test_img[140:340, 0:200])
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
