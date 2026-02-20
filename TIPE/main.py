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
        if isChargement and task_done and (userInput is None or userInput == "n"):
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

# ! ------ Setup UDP Socket ------

PI_IP = "192.168.1.121"
PI_SEND_PORT = 9991
PI_CMD_PORT = 9992

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind(('', PI_SEND_PORT)) 
udp_socket.settimeout(0.1)

print(f"Listening on port 9991 for UDP frames from {PI_IP}:{PI_SEND_PORT}")

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

frame = np.zeros((240,320), dtype=np.uint8)
extradata = b''

try:
    while True:
        affichage.fill((0,0,0))
        
        # Try to receive UDP frame
        try:
            data, addr = udp_socket.recvfrom(65535)
            if len(data) >= struct.calcsize('<LL'):
                header_size = struct.calcsize('<LL')
                image_len, capt_len = struct.unpack('<LL', data[:header_size])
                
                if len(data) >= header_size + image_len + capt_len:
                    img_bytes = data[header_size:header_size + image_len]
                    extradata = data[header_size + image_len:header_size + image_len + capt_len]
                    
                    frame = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    frame = cv2.flip(frame, -1)
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Error receiving frame: {e}")
        
        
        hframe, wframe = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgpg = pg.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        img = frame / 255

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
                    udp_socket.sendto(controlemanuel, (PI_IP, PI_CMD_PORT))
            
#! ------ MODE DETERMINISTE ------
            if mode == 3:
                dc = deterministe.controle(event)
                if dc != None:
                    dctrl, val = dc
                if dctrl == 0: # Stop
                    udp_socket.sendto(b'S', (PI_IP, PI_CMD_PORT))
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
                    if pred in (0,3):
                        udp_socket.sendto(b'C', (PI_IP, PI_CMD_PORT))
                    if pred in (1,2):
                        udp_socket.sendto(b'F', (PI_IP, PI_CMD_PORT))
                    udp_socket.sendto(definition_byte[pred], (PI_IP, PI_CMD_PORT))


#! ------ MODE AUTOMATIQUE ------
        if mode == 0 and not arret and delaycontrol + 0.1 < time.time():
            delaycontrol = time.time() 
            pred = ia_image.clf.predict(img[hframe//2:,:].flatten().reshape(1,-1))
            print(ia_image.definition[pred[0]])
            if pred[0] in (0,3):
                udp_socket.sendto(b'C', (PI_IP, PI_CMD_PORT))
            if pred[0] in (1,2):
                udp_socket.sendto(b'F', (PI_IP, PI_CMD_PORT))
            udp_socket.sendto(definition_byte[pred[0]], (PI_IP, PI_CMD_PORT))

#? ------ Affichage ------
        pg.draw.rect(affichage, (20,20,20), pg.Rect(width/2,0,width/2,height))
        affiche_texte("Flux vidéo:", (255,255,255), width/4, height/2-hframe/2-24, 20, True)
        affichage.blit(imgpg,(width/4 - wframe/2,height/2 - hframe/2))
        affiche_texte(f"Mode: {modedef[mode]}", (255, 255, 255), width/2+20,height/2-hframe/2, 20)
        if mode == 3:
            affiche_texte(f"Prédiction: {ia_image.definition[pred]}", (255,255,255), width/2+20, height/2-hframe/2+22, 20)
        if mode == 0:
            affiche_texte(f"Prédiction: {ia_image.definition[pred[0]]}", (255,255,255), width/2+20, height/2-hframe/2+22, 20)
        if extradata:
            affiche_texte(f"Capteur: {extradata}", (255,255,255), width/2+20, height/2-hframe/2+44, 20)
        pg.display.flip()


finally:
    cv2.destroyAllWindows()
    udp_socket.close()