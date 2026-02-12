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
from modules_pygame.boutons import Button
import threading


# Initialisation de l'affichage

pg.init()
pg.display.set_caption("Le meilleur TIPE de l'UNIVERS")
height = 720
width = 1280
affichage = pg.display.set_mode((width,height))

def affiche_texte(text,text_col,x,y,size,centerx=False,centery=False):
    text_font = pg.font.SysFont("Helvetica", size)
    img = text_font.render(text, True, text_col)
    w,h = text_font.size(text)
    x = x-w/2 if centerx else x
    y = y-h/2 if centery else y
    affichage.blit(img, (x,y))

# ! ------ Ecran de choix du mode de controle ------

isChoixMode = True
mode = 0
modedef = {
    0: "Automatique",
    1: "Manuel",
    2: "Training",
    3: "Déterministe"
}

boutonMode = Button(width//2-175, height//2-40, 350, 80, "Automatique", pg.font.SysFont("Helvetica", 28))
boutonConfirmer = Button(width//2-100, height//2+60, 200, 80, "Confirmer", pg.font.SysFont("Helvetica", 28), (0,225,0), (0,0,0), (255,255,255), (0,200,0))

while isChoixMode:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
    mousepos, isMousePressed = pg.mouse.get_pos(),pg.mouse.get_pressed()[0]
    boutonMode.update(mousepos, isMousePressed)
    boutonConfirmer.update(mousepos, isMousePressed)
    affichage.fill((0,0,0))
    if boutonMode.is_clicked(mousepos, isMousePressed):
        mode = (mode+1)%len(list(modedef.keys()))
        boutonMode.set_text(modedef[mode])
    if boutonConfirmer.is_clicked(mousepos, isMousePressed):
        isChoixMode = False
    
    affiche_texte("Veuillez choisir le mode de controle:", (255,255,255), width//2, height//2-100, 28, True)
    boutonMode.draw(affichage)
    boutonConfirmer.draw(affichage)
    pg.display.flip()

# ! ------ Initialisation ------

isInitialisation = True

result = None
task_done = False

def run_task(func, *args):
    global result, task_done
    result = func(*args)
    task_done = True

isChargement = False
userInput = ""

boutonOui = Button(width//4-100+200, height//2+60, 200, 80, "Oui", pg.font.SysFont("Helvetica", 28), (0,225,0), (0,0,0), (255,255,255), (0,200,0))
boutonNon = Button((3*width)//4-100-200, height//2+60, 200, 80, "Non", pg.font.SysFont("Helvetica", 28), (225,0,0), (0,0,0), (255,255,255), (200,0,0))
boutonLigne = Button(width//4-100+200, height//2+60, 200, 80, "Ligne", pg.font.SysFont("Helvetica", 28))
boutonPoint = Button((3*width)//4-100-200, height//2+60, 200, 80, "Point", pg.font.SysFont("Helvetica", 28))

while isInitialisation:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
        if event.type == pg.KEYDOWN:
            if not isChargement:
                if mode == 0:
                    if event.key == pg.K_BACKSPACE:
                        if len(userInput) > 0:
                            userInput = userInput[:-1]
                    elif event.key == pg.K_RETURN:
                        isChargement = True
                        pred = [3]
                        threading.Thread(
                            target=run_task,
                            daemon=True,
                            args=(ia_image.init,userInput)
                        ).start()
                    else:
                        userInput += event.unicode
            if mode == 3 and isChargement:
                if event.key == pg.K_BACKSPACE:
                    if len(userInput) > 0:
                        userInput = userInput[:-1]
                elif event.key == pg.K_RETURN:
                    isInitialisation = False
                    seuil = float(userInput)
                else:
                    userInput += event.unicode
    
    mousepos, isMousePressed = pg.mouse.get_pos(),pg.mouse.get_pressed()[0]
    affichage.fill((0,0,0))

    if mode == 0:
        if not isChargement:
            ia_image.texte = "Numéro du modèle à charger (n: nouveau par défaut): "
            affiche_texte(userInput, (255,255,255), width/2, height/2-10, 28, True)
        if isChargement and not task_done:
            affiche_texte("Chargement...", (255,255,255), width/2, height/2-70, 28, True)
        if isChargement and task_done:
            affiche_texte("Sauvegarder le modèle ? (o/n):", (255,255,255), width/2, height/2-10, 28, True)
            boutonOui.update(mousepos, isMousePressed)
            boutonOui.draw(affichage)
            boutonNon.update(mousepos, isMousePressed)
            boutonNon.draw(affichage)
            if boutonOui.is_clicked(mousepos,isMousePressed):
                ia_image.clf = result
                ia_image.save_model(ia_image.clf)
                isInitialisation = False
            if boutonNon.is_clicked(mousepos,isMousePressed):
                ia_image.clf = result
                isInitialisation = False
        affiche_texte(ia_image.texte, (255,255,255), width/2, height/2-40, 28, True)
    
    if mode == 2:
        boutonOui.update(mousepos, isMousePressed)
        boutonOui.draw(affichage)
        boutonNon.update(mousepos, isMousePressed)
        boutonNon.draw(affichage)
        if not isChargement:
            affiche_texte("Supprimer les anciennes images d'apprentissage ? (o/n):", (255,255,255), width/2, height/2-40, 28, True)
            if boutonOui.is_clicked(mousepos,isMousePressed):
                isChargement = True
            if boutonNon.is_clicked(mousepos,isMousePressed):
                i = training.compte_fichiers()
                isInitialisation = False
        if isChargement:
            affiche_texte("Êtes-vous sûr de supprimer les images ? (Cette action est IRREVERSIBLE !) (o/n):", (255,255,255), width/2, height/2-40, 28, True)
            if boutonOui.is_clicked(mousepos,isMousePressed):
                training.supprime_fichiers()
                i = 0
                isInitialisation = False
            if boutonNon.is_clicked(mousepos,isMousePressed):
                i = training.compte_fichiers()
                isInitialisation = False
    
    if mode == 3:
        pred = 3
        if not isChargement:
            affiche_texte('Choisissez la version du mode déterministe:', (255,255,255), width/2, height/2-44, 28, True)
            boutonPoint.update(mousepos, isMousePressed)
            boutonPoint.draw(affichage)
            boutonLigne.update(mousepos, isMousePressed)
            boutonLigne.draw(affichage)
            if boutonLigne.is_clicked(mousepos,isMousePressed):
                modedet = 1
                isChargement = True
            if boutonPoint.is_clicked(mousepos,isMousePressed):
                modedet = 2
                isChargement = True
        if isChargement:
            affiche_texte('Choisissez le seuil de détection des lignes blanches (ex: 0.8):', (255,255,255), width/2, height/2-44, 28, True)
            affiche_texte(userInput, (255,255,255), width/2, height/2, 28, True, True)

    if mode == 1:
        isInitialisation = False

    pg.display.flip()

# ! ------ Connection à la Raspberry Pi ------

HOST = '' if len(sys.argv) < 2 else sys.argv[1]
PORT = 9991 if len(sys.argv) < 3 else int(sys.argv[2])

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))

connected = False
connected_time = 0

conn, addr = None,None

state = 0

def wait_for_connection():
    global connected, connected_time, conn, addr
    server_socket.listen(0)
    print(f"Ecoute {HOST}:{PORT}...")
    conn, addr = server_socket.accept()
    print("Connecté par", addr)
    connected = True
    connected_time = pg.time.get_ticks()

# ? threading permet d'executer la fonction en même temps que la boucle
threading.Thread(
    target=wait_for_connection,
    daemon=True
).start()

while state!=2:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
    current_time = pg.time.get_ticks()
    if state == 0:
        affichage.fill((0, 0, 0))
        affiche_texte("En attente de connection...", (255,255,255), width//2, height//2, 32, True, True)

        if connected:
            state = 1

    elif state == 1:
        affichage.fill((0, 0, 0))
        affiche_texte("Connecté!", (0,255,0), width//2, height//2, 32, True, True)

        if current_time - connected_time >= 2000:
            state = 2
    pg.display.flip()

connection = conn.makefile('rb')
prev_time = time.time()

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

definition_byte = [b'F', b'R', b'L', b'S']

dctrl = None
val = False

# ! ------ Boucle Principale ------ 

try:
    while True:
        affichage.fill((0,0,0))

        header = connection.read(struct.calcsize('<LL'))
        if not header:
            break
        image_len, capt_len = struct.unpack('<LL', header)
        if not image_len or not capt_len:
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

        extradata = b''
        while len(extradata) < capt_len:
            more = connection.read(capt_len - len(extradata))
            if not more:
                break
            extradata += more

#? ------ Boutons de controle pygame ------
        for event in pg.event.get():
            if event.type == pg.QUIT: # Ferme la fenêtre
                pg.quit()
                sys.exit()

#! ------ MODE TRAINING ------
            if mode == 2:
                i += training.main(frame, event, i) # i est le nombre de photos
                    
#! ------ MODE MANUEL ------
            if mode == 1 or mode == 2:
                controlemanuel = manuel.main(event)
                if controlemanuel != b'':
                    conn.sendall(manuel.main(event))
            
#! ------ MODE DETERMINISTE ------
            if mode == 3:
                dc = deterministe.controle(event)
                if dc != None:
                    dctrl, val = dc
                if dctrl == 0: # Stop
                    conn.sendall(b'S')
                    arret = True
                else:
                    controle_d[dctrl] = val
        
        if mode == 3:
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


#! ------ MODE AUTOMATIQUE ------
        if mode == 0 and not arret and delaycontrol + 0.1 < time.time():
            delaycontrol = time.time() 
            pred = ia_image.clf.predict(img[hframe//2:,:].flatten().reshape(1,-1))
            print(ia_image.definition[pred[0]])
            if pred[0] in (0,3):    # Centre les roues pour avancer ou reculer
                conn.sendall(b'C')
            if pred[0] in (1,2):    # Avance les roues pour droite et gauche
                conn.sendall(b'F')
            conn.sendall(definition_byte[pred[0]])

#? ------ Affichage ------
        pg.draw.rect(affichage, (20,20,20), pg.Rect(width/2,0,width/2,height))
        affiche_texte("Flux vidéo:", (255,255,255), width/4, height/2-hframe/2-24, 20, True)
        affichage.blit(imgpg,(width/4 - wframe/2,height/2 - hframe/2))
        affiche_texte(f"Mode: {modedef[mode]}", (255, 255, 255), width/2+20,height/2-hframe/2, 20)
        if mode == 3:
            affiche_texte(f"Prédiction: {ia_image.definition[pred]}", (255,255,255), width/2+20, height/2-hframe/2+22, 20)
        if mode == 0:
            affiche_texte(f"Prédiction: {ia_image.definition[pred[0]]}", (255,255,255), width/2+20, height/2-hframe/2+22, 20)
        affiche_texte(f"Capteur: {extradata}", (255,255,255), width/2+20, height/2-hframe/2+44, 20)
        pg.display.flip() # Met à jour la fenêtre pygame



finally:
    cv2.destroyAllWindows()
    connection.close()
    conn.close()
    server_socket.close()


