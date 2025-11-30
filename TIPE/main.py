import socket
import struct
import cv2
import numpy as np
import sys
import time
import pygame as pg
import ia_image
import deterministe
import training
import manuel

# Choix du mode de controle ou training
mode = input("Veuillez choisir le mode de controle \n(Automatique: a, Manuel: m, Training: t, Déterministe: d): ").lower()

# Choix du modèle à charger
if mode == "a":
    ia_image.clf = ia_image.init()

# Possibilité de supprimer les anciennes images en mode training
if mode == "t":
    training.init()


# Connection au socket

HOST = '' if len(sys.argv) < 2 else sys.argv[1]
PORT = 9998 if len(sys.argv) < 3 else int(sys.argv[2])

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(0)
print(f"Ecoute {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print("Connecté par", addr)

connection = conn.makefile('rb')
prev_time = time.time()


# Initialisation de l'affichage

pg.init()
pg.display.set_caption("Le meilleur TIPE de l'UNIVERS")
affichage = pg.display.set_mode((320,240))

hauteur = 120 # Hauteur pour le mode déterministe
delayline = time.time()

controle_d = {     # Controle de la ligne du mode déterministe
    "down": False,
    "up": False
}

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
        if mode == "t": 
            training.main(frame)
                    
#! ---- MODE MANUEL ----
        if mode == "m":
            conn.sendall(manuel.main())
            
#! ---- MODE DETERMINISTE ----
        if mode == "d":
            dctrl, val = deterministe.controle()
            if dctrl == 0: #Stop
                conn.sendall(b'S')
                arret = True
            else:
                controle_d[dctrl] = val

            if not arret:
                pg.draw.line(affichage, (0,255,0), (0,hauteur), (320,hauteur))
                if controle_d["down"] and time.time() > delayline + 0.02 and hauteur < 240 - 1:
                    delayline = time.time()
                    hauteur+=1
                    print(hauteur)
                if controle_d["up"] and time.time() > delayline + 0.02 and hauteur > 0:
                    delayline = time.time()
                    hauteur-=1
                    print(hauteur)
                pred = deterministe.predDet(img, hauteur)
                print(ia_image.definition[pred])
                if pred in (0,3):    # Centre les roues pour avancer ou reculer
                    conn.sendall(b'C')
                if pred in (1,2):    # Avance les roues pour droite et gauche
                    conn.sendall(b'F')
                conn.sendall(definition_byte[pred])


#! ---- MODE AUTOMATIQUE ----
            if mode == "a" and not arret: 
                pred = ia_image.clf.predict(img[120:,:].flatten().reshape(1,-1))
                print(ia_image.definition[pred[0]])
                if pred[0] in (0,3):    # Centre les roues pour avancer ou reculer
                    conn.sendall(b'C')
                if pred[0] in (1,2):    # Avance les roues pour droite et gauche
                    conn.sendall(b'F')
                conn.sendall(definition_byte[pred[0]])


        pg.display.flip() # Met à jour la fenêtre pygame


finally:
    cv2.destroyAllWindows()
    connection.close()
    conn.close()
    server_socket.close()



