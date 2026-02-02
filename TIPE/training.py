import shutil
import os
import pygame as pg
import cv2

texte = ""

def supprime_fichiers():
    dossiers = ["training_data/avance", "training_data/droite", "training_data/gauche", "training_data/recule"]
    for dossier in dossiers:
        shutil.rmtree(dossier)
        os.makedirs(dossier)
    print("Anciennes images supprimées !")

def compte_fichiers():
    dossiers = ["training_data/avance", "training_data/droite", "training_data/gauche", "training_data/recule"]
    i = 0
    for dossier in dossiers:
        i += len(os.listdir(dossier))
    return i

def main(frame, event, i):
    if event.type == pg.KEYDOWN:
        if event.key == pg.K_UP:
            cv2.imwrite(f"training_data/avance/{i}.jpg", frame[120:,:])
            return 1
        if event.key == pg.K_RIGHT:
            cv2.imwrite(f"training_data/droite/{i}.jpg", frame[120:,:])
            return 1
        if event.key == pg.K_LEFT:
            cv2.imwrite(f"training_data/gauche/{i}.jpg", frame[120:,:])
            return 1
        if event.key == pg.K_DOWN:
            cv2.imwrite(f"training_data/recule/{i}.jpg", frame[120:,:])
            return 1
        if event.key == pg.K_t:
            cv2.imwrite(f"test/img_test.jpg", frame[120:,:])
            return 1
    return 0