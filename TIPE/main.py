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
    pred = [3]
    ia_image.clf = ia_image.init()

# Possibilité de supprimer les anciennes images en mode training
if mode == "t":
    i = training.init()

if mode == "d":
    pred = 3
    modedet,seuil = deterministe.init()

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
height = 240
width = 640
affichage = pg.display.set_mode((width,height))

text_font = pg.font.SysFont("Arial", 18)

def affiche_texte(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    affichage.blit(img, (x,y))

hauteur = 200 # Hauteur pour le mode déterministe
ecartement = 100 # Ecartement pour le mode déterministe
delayline = time.time()
delaycontrol = delayline

controle_d = {     # Controle de la ligne du mode déterministe
    "down": False,
    "up": False,
    "right": False,
    "left": False
}

arret = False # Arrêt de la voiture

definition_byte = [b'F', b'R', b'L', b'B']

dctrl = None
val = False

try:
    while True:
        affichage.fill((0,0,0))

        image_len_data = connection.read(struct.calcsize('<L'))
        if not image_len_data:
            break
        image_len = struct.unpack('<L', image_len_data)[0]
        if not image_len:
            break

        # * Lecture des données de l'image
        data = b''
        while len(data) < image_len:
            more = connection.read(image_len - len(data))
            if not more:
                break
            data += more

        # * Decode les bytes en image
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        frame = cv2.flip(frame,-1)
        hframe, wframe = frame.shape

        # * Transforme l'image en surface pygame pour afficher le flux vidéo
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        imgpg = pg.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        
        img = frame/255

#? ---- Boutons de controle pygame ----
        for event in pg.event.get():
            if event.type == pg.QUIT: # Ferme la fenêtre
                pg.quit()
                sys.exit()

#! ---- MODE TRAINING ----
            # TODO Commentaires à enlever si ne pose pas de problème
            if mode == "t": # and delaycontrol + 0.1 < time.time():
                # delaycontrol = time.time()
                i += training.main(frame, event, i) # i est le nombre de photos
                    
#! ---- MODE MANUEL ----
            if mode == "m": # and delaycontrol + 0.1 < time.time():
                # delaycontrol = time.time()
                controlemanuel = manuel.main(event)
                if controlemanuel != b'':
                    conn.sendall(manuel.main(event))
            
#! ---- MODE DETERMINISTE ----
            if mode == "d":
                dc = deterministe.controle(event)
                if dc != None:
                    dctrl, val = dc
                if dctrl == 0: # Stop
                    conn.sendall(b'S')
                    arret = True
                else:
                    controle_d[dctrl] = val
        
        if mode == "d":
            if not arret:
                if modedet == 1:
                    pg.draw.line(imgpg, (0,255,0), (0,hauteur), (wframe,hauteur))
                if modedet == 2:
                    pg.draw.circle(imgpg, (0,255,0), (ecartement,hauteur), 3)
                    pg.draw.circle(imgpg, (0,255,0), (wframe-ecartement,hauteur), 3)
                dmain = deterministe.main(delayline, controle_d, hauteur, ecartement, wframe, hframe)

                if dmain != None:
                    delayline, hauteur, ecartement = dmain
                
                if delaycontrol + 0.1 < time.time():
                    delaycontrol = time.time()
                    npred = deterministe.predDet(img, hauteur, ecartement, seuil, modedet)
                    if npred != None:
                        pred = npred
                    print(ia_image.definition[pred])
                    if pred in (0,3):    # Centre les roues pour avancer ou reculer
                        conn.sendall(b'C')
                    if pred in (1,2):    # Avance les roues pour droite et gauche
                        conn.sendall(b'F')
                    conn.sendall(definition_byte[pred])


#! ---- MODE AUTOMATIQUE ----
        if mode == "a" and not arret and delaycontrol + 0.1 < time.time():
            delaycontrol = time.time() 
            pred = ia_image.clf.predict(img[hframe//2:,:].flatten().reshape(1,-1))
            print(ia_image.definition[pred[0]])
            if pred[0] in (0,3):    # Centre les roues pour avancer ou reculer
                conn.sendall(b'C')
            if pred[0] in (1,2):    # Avance les roues pour droite et gauche
                conn.sendall(b'F')
            conn.sendall(definition_byte[pred[0]])

#? ---- Affichage ----
        affichage.blit(imgpg,(0,0))
        affiche_texte(f"Mode: {mode}", text_font, (255, 255, 255), wframe+2,2)
        if mode == "d":
            affiche_texte(f"Prédiction: {ia_image.definition[pred]}", text_font, (255,255,255), wframe+2, 20)
        if mode == "a":
            affiche_texte(f"Prédiction: {ia_image.definition[pred[0]]}", text_font, (255,255,255), wframe+2, 20)
        pg.display.flip() # Met à jour la fenêtre pygame


finally:
    cv2.destroyAllWindows()
    connection.close()
    conn.close()
    server_socket.close()



