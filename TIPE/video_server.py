import socket
import struct
import cv2
import numpy as np
import sys
import time
import pygame as pg
import ia_image
from deterministe import predDet
import os
import shutil

mode = input("Veuillez choisir le mode de controle \n(Automatique: a, Manuel: m, Training: t, Déterministe: d): ").lower()

# Demande si on veut un nouveau modèle
if mode == "a":
    model = input("Numéro du modèle à charger (n: nouveau, 0 par défaut): ")
    if model.lower() == 'n':
        ia_image.clf = ia_image.train()
    else:
        if model.isdigit():
            ia_image.clf = ia_image.load_model(int(model))
        else:
            ia_image.clf = ia_image.load_model()

# Possibilité de supprimer les anciennes images
if mode == "t":
    supprimer = input("Supprimer les anciennes images de training? (o/n): ").lower()
    if supprimer == 'o':
        dossiers = ["training_data/avance", "training_data/droite", "training_data/gauche", "training_data/recule"]
        for dossier in dossiers:
            shutil.rmtree(dossier)
            os.makedirs(dossier)
        print("Anciennes images supprimées.")
    i = 0 # numéro d'image

HOST = '' if len(sys.argv) < 2 else sys.argv[1]
PORT = 9998 if len(sys.argv) < 3 else int(sys.argv[2])

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(0)
print(f"Listening on {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print("Connected by", addr)

connection = conn.makefile('rb')
prev_time = time.time()

pg.init()
pg.display.set_caption("Le meilleur TIPE de l'UNIVERS")
affichage = pg.display.set_mode((320,240))

hauteur = 120 # Hauteur pour le mode déterministe
delayline = time.time()
delaycontrol = time.time()

down = False # Controle de la ligne du mode déterministe
up = False

arret = False # Arrêt de la voiture

definition_byte = [b'F', b'R', b'L', b'B']

try:
    while True:
        image_len_data = connection.read(struct.calcsize('<L'))
        if not image_len_data:
            break
        image_len = struct.unpack('<L', image_len_data)[0]
        if not image_len:
            break

        # Lecture des données de l'image
        data = b''
        while len(data) < image_len:
            more = connection.read(image_len - len(data))
            if not more:
                break
            data += more

        # Decode les bytes en image
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        frame = cv2.flip(frame,-1)

        # Afficher le flux vidéo
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        imgpg = pg.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        affichage.blit(imgpg,(0,0))

        
        img = frame/255


#! ---- MODE TRAINING ----
        if mode == "t" and delaycontrol + 0.1 < time.time():
            delaycontrol = time.time() 
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_UP:
                        cv2.imwrite(f"training_data/avance/{i}.jpg", frame[120:,:])
                    if event.key == pg.K_RIGHT:
                        cv2.imwrite(f"training_data/droite/{i}.jpg", frame[120:,:])
                    if event.key == pg.K_LEFT:
                        cv2.imwrite(f"training_data/gauche/{i}.jpg", frame[120:,:])
                    if event.key == pg.K_DOWN:
                        cv2.imwrite(f"training_data/recule/{i}.jpg", frame[120:,:])
                    if event.key == pg.K_t:
                        cv2.imwrite(f"test/img_test.jpg", frame[120:,:])
                    i+=1
            


        if mode != "t":
            for event in pg.event.get(): 
                if event.type == pg.KEYDOWN:    #Possibilité d'arrêter la voiture même en mode Automatique
                    if event.key == pg.K_SPACE:
                        conn.sendall(b'S')
                        print("Stop")
                        arret = True
                    
#! ---- MODE MANUEL ----
                    if mode == "m" and delaycontrol + 0.1 < time.time():
                        delaycontrol = time.time()                  
                        if event.key == pg.K_z:      # forward
                            conn.sendall(b'F')
                            print("Forward")
                        elif event.key == pg.K_s:    # backward
                            conn.sendall(b'B')
                            print("Backward")
                        elif event.key == pg.K_q:    # left
                            conn.sendall(b'L')
                            print("Left")
                        elif event.key == pg.K_d:    # right
                            conn.sendall(b'R')
                            print("Right")
        
                elif event.type == pg.KEYUP and mode == "m":
                    if event.key == pg.K_z or event.key == pg.K_s:
                    # stop the car when key is released
                        conn.sendall(b'S')
                    if event.key == pg.K_q or event.key == pg.K_d:
                        conn.sendall(b'C')
            
#! ---- MODE DETERMINISTE ----
                if event.type == pg.KEYDOWN and mode == "d":
                    if event.key == pg.K_UP:
                        up = True
                    if event.key == pg.K_DOWN:
                        down = True
                if event.type == pg.KEYUP and mode == "d":
                    if event.key == pg.K_UP:
                        up = False
                    if event.key == pg.K_DOWN:
                        down = False
                elif event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()

            if mode == "d" and arret == False:
                pg.draw.line(affichage, (0,255,0), (0,hauteur), (320,hauteur))
                if down and time.time() > delayline + 0.001 and hauteur < 240 - 1:
                    delayline = time.time()
                    hauteur+=1
                    print(hauteur)
                if up and time.time() > delayline + 0.001 and hauteur > 0:
                    delayline = time.time()
                    hauteur-=1
                    print(hauteur)
                if delaycontrol + 0.1 < time.time():
                    delaycontrol = time.time()
                    pred = predDet(img, hauteur)
                    print(ia_image.definition[pred])
                    if pred in (0,3):
                        conn.sendall(b'C')
                    if pred in (1,2):
                        conn.sendall(b'F')
                    conn.sendall(definition_byte[pred])


#! ---- MODE AUTOMATIQUE ----
            if mode == "a" and arret == False and delaycontrol + 0.1 < time.time():
                delaycontrol = time.time() 
                pred = ia_image.clf.predict(img[120:,:].flatten().reshape(1,-1))
                print(ia_image.definition[pred[0]])
                if pred[0] in (0,3):
                    conn.sendall(b'C')
                if pred[0] in (1,2):
                    conn.sendall(b'F')
                conn.sendall(definition_byte[pred[0]])


        pg.display.flip() # Met à jour la fenêtre pygame


finally:
    cv2.destroyAllWindows()
    connection.close()
    conn.close()
    server_socket.close()


