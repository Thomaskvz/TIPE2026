import pygame as pg
import time

def predDet(img, hauteur, ecartement, mode=1):
    if mode == 1:
        blanc = 0
        while blanc < len(img[0]) and img[hauteur,blanc] < 0.8:
            blanc += 1
        if blanc == len(img[0]):        # Pas de blanc: Recule
            return 3
        if blanc >= len(img[0])//2:     # Blanc à droite: Droite
            return 2
        blanc = len(img[0]) -1
        while blanc < len(img[0]) and img[hauteur,blanc] < 0.8:
            blanc -= 1
        if blanc < len(img[0])//2:      # Blanc à gauche: Gauche
            return 1
        else:                           # Blanc aux 2: Avance
            return 0
    
    if mode == 2:
        if img[hauteur, ecartement] > 0.7 and img[hauteur, len(img)-ecartement] > 0.7:
            return 0
        if img[hauteur, ecartement] > 0.7:
            return 1
        if img[hauteur, len(img)-ecartement] > 0.7:
            return 2
        else:
            return 3

def controle(event):
    if event.type == pg.KEYDOWN:    #Possibilité d'arrêter la voiture même en mode Automatique
        if event.key == pg.K_SPACE:
            print("Stop")
            return 0
        if event.key == pg.K_UP:
            return "up", True
        if event.key == pg.K_DOWN:
            return "down", True
        if event.key == pg.K_RIGHT:
            return "right", True
        if event.key == pg.K_LEFT:
            return "left", True
    if event.type == pg.KEYUP:
        if event.key == pg.K_UP:
            return "up", False
        if event.key == pg.K_DOWN:
            return "down", False
        if event.key == pg.K_RIGHT:
            return "right", False
        if event.key == pg.K_LEFT:
            return "left", False
        
def main(delayline, controle_d, hauteur, ecartement, width, height):
    curtime = time.time()
    if controle_d["down"] and curtime > delayline + 0.001 and hauteur < height - 1:
        hauteur+=1
        return curtime, hauteur, ecartement
    if controle_d["up"] and curtime > delayline + 0.001 and hauteur > 0:
        hauteur-=1
        return curtime, hauteur, ecartement
    if controle_d["right"] and curtime > delayline + 0.001 and ecartement < width//2 - 1:
        ecartement+=1
        return curtime, hauteur, ecartement
    if controle_d["left"] and curtime > delayline + 0.001 and ecartement > 0:
        ecartement-=1
        return curtime, hauteur, ecartement

    
def init():
    mode = input('Choisissez la version du mode déterministe (p: point ou l: ligne): ')
    if mode == "p":
        return 2
    return 1


if __name__ == "__main__":
    import cv2
    import numpy as np

    dossier = "training_data/gauche"
    nom = "156"
    test_img = cv2.imread(f"{dossier}/{nom}.jpg", cv2.IMREAD_GRAYSCALE)
    test_img = test_img / 255


    print(test_img[240, 50], test_img[240, 540])
    prediction = predDet(test_img, 120, 50, 1)
    definition = ["Avance", "Droite", "Gauche", "Recule"]
    print(f"Prédiction déterministe: {definition[prediction]}")



    # while True:
    #     cv2.imshow("Test Image", test_img[140:340, 0:200])
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
