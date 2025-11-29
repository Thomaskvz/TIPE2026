import pygame as pg
import sys

def main():
    for event in pg.event.get(): 
        if event.type == pg.KEYDOWN:    #Possibilité d'arrêter la voiture même en mode Automatique
            if event.key == pg.K_SPACE:
                print("Stop")     
                return b'S'
            if event.key == pg.K_z:      # forward
                print("Forward")
                return b'F'
            elif event.key == pg.K_s:    # backward
                print("Backward")
                return b'B'
            elif event.key == pg.K_q:    # left
                print("Left")
                return b'L'
            elif event.key == pg.K_d:    # right
                print("Right")
                return b'R'

        elif event.type == pg.KEYUP:
            if event.key == pg.K_z or event.key == pg.K_s:
            # stop the car when key is released
                return b'S'
            if event.key == pg.K_q or event.key == pg.K_d:
                return b'C'
            
        elif event.type == pg.QUIT:
            pg.quit()
            sys.exit()
