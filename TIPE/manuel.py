import pygame as pg
import sys

def main(event):
    if event.type == pg.KEYDOWN:    #Possibilité d'arrêter la voiture même en mode Automatique
        if event.key == pg.K_SPACE:
            print("Stop")     
            return b'S'
        if event.key == pg.K_z:
            print("Avance")
            return b'F'
        elif event.key == pg.K_s:
            print("Recule")
            return b'B'
        elif event.key == pg.K_q:
            print("Gauche")
            return b'L'
        elif event.key == pg.K_d:
            print("Droite")
            return b'R'

    elif event.type == pg.KEYUP:
        if event.key == pg.K_z or event.key == pg.K_s:
        # Arrête la voiture lorsque la touche est levée
            return b'S'
        if event.key == pg.K_q or event.key == pg.K_d:
            return b'C'
    return b''
