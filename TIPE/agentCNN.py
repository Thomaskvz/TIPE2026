from matplotlib.pyplot import pause
import random
import numpy as np
from collections import deque
from environmentCNN import Environment
from modelCNN import Conv_QNet, QTrainer
import time
import os
import csv
import pygame as pg
import cv2
import sys
from modules_pygame.boutons import Button
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

MAX_MEMORY = 100_000
BATCH_SIZE = 64
LR = 0.001
EPS_DECAY = 1000
EPS_START = 0.99
DELAI_ACTIONS = 0.4
NB_RETOURS = 3
NEURONES_CACHE1 = 512
NEURONES_CACHE2 = 256

# ! ------ Initialisation Pygame ------
pg.init()
pg.display.set_caption("Agent DQN Training")
height = 720
width = 1280
affichage = pg.display.set_mode((width, height))
clock = pg.time.Clock()

def affiche_texte(text, text_col, x, y, size, centerx=False, centery=False):
    text_font = pg.font.SysFont("Helvetica", size)
    img = text_font.render(text, True, text_col)
    w, h = text_font.size(text)
    x = x - w/2 if centerx else x
    y = y - h/2 if centery else y
    affichage.blit(img, (x, y))

class Agent():
    def __init__(self, input_height, input_width, num_actions=3):
        self.epsilon = 0
        self.gamma = 0.99
        self.memory = deque(maxlen=MAX_MEMORY)
        
        # On instancie le nouveau modèle CNN (Plus besoin des paramètres de neurones cachés)
        self.model = Conv_QNet(num_actions)
        self.trainer = QTrainer(self.model, LR, self.gamma)
        self.steps_done = 0
        self.input_height = input_height
        self.input_width = input_width

    def get_state(self, env):
        flattened_img = env.process_image()[0]
        # On reconstruit l'image en 2D et on ajoute la dimension du canal (1, Hauteur, Largeur)
        return flattened_img.reshape(1, self.input_height, self.input_width)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_memory(self, nb, nb_actions=0):
        L = []
        debut = max(len(self.memory)-nb_actions, len(self.memory)-nb)
        for i in range(debut, len(self.memory)):
            L.append(self.memory[i][1])
        return L[::-1]

    def get_action(self, state):
        # Utilisation d'un decay exponentiel propre basé sur le nombre d'actions totales
        self.epsilon = max(0.01, EPS_START * (0.9995 ** self.steps_done))
        
        if random.random() < self.epsilon:
            action = random.randint(0, 2)
        else:
            # state a déjà la forme (1, 36, 80). On ajoute la dimension de Batch -> (1, 1, 36, 80)
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            pred = self.model(state0)
            action = torch.argmax(pred).item()
            
        return action

temps_episodes = []
temps_episodes_moyen = []
temps_total = 0
record = 0
numModel = len(os.listdir("./models"))

hframe, wframe = 60, 80
debutimg = hframe//3 + 4
input_height = hframe - debutimg


agent = Agent(input_height, wframe, num_actions=3)
env = Environment(debutimg)
env.reset()

episode = 0

boutonCommencer = Button(3*width//4 - 100, height//2, 200, 80, "Commencer", pg.font.SysFont("Helvetica", 28), (0, 225, 0), (0, 0, 0), (255, 255, 255), (0, 200, 0))
boutonArreter = Button(3*width//4 - 100, height//2 + 100, 200, 80, "Arrêter", pg.font.SysFont("Helvetica", 28), (225, 0, 0), (0, 0, 0), (255, 255, 255), (200, 0, 0))

temps = 0
action = 0
record = 0

arret_episode = False

userInput = ""
debut = True
# ! ------ Modifier une IA pré-entraînée ------
while debut:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_BACKSPACE:
                if len(userInput) > 0:
                    userInput = userInput[:-1]
            elif event.key == pg.K_RETURN:
                debut = False
            else:
                userInput += event.unicode
    affichage.fill((0, 0, 0))
    affiche_texte("Entrez le numéro du modèle à charger (ou laissez vide pour un nouveau modèle):", (255, 255, 255), width//2, height//2 - 50, 24, centerx=True)
    affiche_texte(userInput, (255, 255, 255), width//2, height//2, 28, centerx=True)
    pg.display.flip()

if userInput != "":
    numModel = userInput
    pathmodel = os.path.join("./models", f"model_dqn{numModel}.pth")
    if os.path.exists(pathmodel):
        with open(os.path.join("./resultats", f"{numModel}.txt"), "r") as f:
            L = f.readlines()
        L = [line.strip().split(": ") for line in L]
        print(L)
        agent = Agent(input_height, wframe, num_actions=3)
        agent.gamma = float(L[1][1])
        agent.trainer.lr = float(L[2][1])
        BATCH_SIZE = int(L[3][1])
        EPS_DECAY = int(L[6][1])
        DELAI_ACTIONS = float(L[7][1])
        NB_RETOURS = int(L[8][1])
        agent.steps_done = int(L[9][1])

        agent.model.load_state_dict(torch.load(pathmodel, weights_only=True))
        print(f"Modèle chargé: model_dqn{numModel}.pth")
    else:
        print(f"Aucun modèle trouvé avec le numéro {numModel}. Un nouveau modèle sera entraîné.")

# ! ------ Passage d'episode manuel ou automatique ? ------
choixManuel = True
isManuel = True

boutonOui = Button(width//4 - 100, height//2 - 40, 200, 80, "Oui", pg.font.SysFont("Helvetica", 28), (0, 225, 0), (0, 0, 0), (255, 255, 255), (0, 200, 0))
boutonNon = Button(3*width//4 - 100, height//2 - 40, 200, 80, "Non", pg.font.SysFont("Helvetica", 28), (225, 0, 0), (0, 0, 0), (255, 255, 255), (200, 0, 0))

while choixManuel:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
                isManuel = False
                choixManuel = False
    affichage.fill((0, 0, 0))
    affiche_texte("Voulez-vous passer les épisodes automatiquement ?", (255, 255, 255), width//2, height//2 - 50, 28, centerx=True)
    
    mousepos, isMousePressed = pg.mouse.get_pos(), pg.mouse.get_pressed()[0]

    boutonOui.update(mousepos, isMousePressed)
    boutonNon.update(mousepos, isMousePressed)
    boutonOui.draw(affichage)
    boutonNon.draw(affichage)
    
    if boutonOui.is_clicked(mousepos, isMousePressed):
        isManuel = False
        choixManuel = False
    
    if boutonNon.is_clicked(mousepos, isMousePressed):
        isManuel = True
        choixManuel = False
    pg.display.flip()

isManuelChoisi = isManuel

with open(f"./resultats/{numModel}.csv", "w", newline='') as f:
    file = csv.writer(f)
    file.writerow(["Episode", "Nombre d'actions réalisées", "Nombre d'action moyen", "Record"])



try:
    while True:
        # ! ------ Écran d'attente avant l'épisode ------
        print(f"isManuel: {isManuel}, isManuelChoisi: {isManuelChoisi}")
        arretAuto = False
        if not isManuel:
            start_time = time.time()
            while time.time() - start_time < 1.0:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                        sys.exit()
                affichage.fill((0, 0, 0))
                pg.draw.rect(affichage, (20, 20, 20), pg.Rect(width//2, 0, width//2, height))

                frame, sensor = env.process_image()
                # print(sensor)
                # if frame is not None:
                #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                #     imgpg = pg.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                #     imgmoit = pg.surfarray.make_surface(frame_rgb[debutimg:, :].swapaxes(0, 1))
                #     imgpg = pg.transform.scale(imgpg, (320, 240))
                #     imgmoit = pg.transform.scale(imgmoit, (320, 120))
                #     wframe, hframe = imgpg.get_size()

                #     affiche_texte("Flux vidéo:", (255, 255, 255), width//4, height//2 - hframe - 48, 20, True)
                #     affichage.blit(imgpg, (width//4 - wframe//2, height//2 - hframe - 24))

                #     affiche_texte("Flux traité:", (255, 255, 255), width//4, height//2 + hframe//2 - 24, 20, True)
                #     affichage.blit(imgmoit, (width//4 - wframe//2, height//2 + hframe//2))
                
                affiche_texte(f"Épisode: {episode}", (255, 255, 255), width//2 + 20, height//2 - hframe//2, 20)
                affiche_texte(f"Nb d'Actions: {temps}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 22, 20)
                affiche_texte(f"Action: {action}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 44, 20)
                affiche_texte(f"Record: {record}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 66, 20)
                affiche_texte(f"Epsilon: {agent.epsilon:.4f}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 88, 20)
                affiche_texte(f"Retour: {agent.get_memory(5, temps)}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 110, 20)
                affiche_texte("Prochain épisode dans 1s...", (255, 255, 255), 3*width//4, height//2 - 170, 28, True)
                
                boutonArreter.update(pg.mouse.get_pos(), pg.mouse.get_pressed()[0])
                boutonArreter.draw(affichage)
                
                if boutonArreter.is_clicked(pg.mouse.get_pos(), pg.mouse.get_pressed()[0]):
                    isManuel = True
                    break

                if sensor == b'01' or sensor == b'10' or sensor == b'11':
                    affiche_texte("Voiture mal replacée!", (255, 0, 0), width//2, height//2 + 100, 28, centerx=True)
                    isManuel = True
                    time.sleep(1.0)
                    break

                pg.display.flip()

        if isManuel:
            waiting = True
            while waiting:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                        sys.exit()
                
                mousepos, isMousePressed = pg.mouse.get_pos(), pg.mouse.get_pressed()[0]
                affichage.fill((0, 0, 0))
                pg.draw.rect(affichage, (20, 20, 20), pg.Rect(width//2, 0, width//2, height))
                
                env.process_image()
                frame = env.get_frame()
                hframe, wframe = frame.shape
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    imgpg = pg.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                    imgmoit = pg.surfarray.make_surface(frame_rgb[debutimg:, :].swapaxes(0, 1))
                    imgpg = pg.transform.scale(imgpg, (320, 240))
                    imgmoit = pg.transform.scale(imgmoit, (320, 120))
                    wframe, hframe = imgpg.get_size()

                    affiche_texte("Flux vidéo:", (255, 255, 255), width//4, height//2 - hframe - 48, 20, True)
                    affichage.blit(imgpg, (width//4 - wframe//2, height//2 - hframe - 24))

                    affiche_texte("Flux traité:", (255, 255, 255), width//4, height//2 + hframe//2 - 24, 20, True)
                    affichage.blit(imgmoit, (width//4 - wframe//2, height//2 + hframe//2))
                
                affiche_texte(f"Épisode: {episode}", (255, 255, 255), width//2 + 20, height//2 - hframe//2, 20)
                affiche_texte(f"Nb d'Actions: {temps}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 22, 20)
                affiche_texte(f"Action: {action}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 44, 20)
                affiche_texte(f"Record: {record}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 66, 20)
                affiche_texte(f"Epsilon: {agent.epsilon:.4f}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 88, 20)
                affiche_texte(f"Retour: {agent.get_memory(NB_RETOURS, temps)}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 110, 20)
                affiche_texte("Veuillez replacer la voiture...", (255, 255, 255), 3*width//4, height//2 - 170, 28, True)
                
                boutonCommencer.update(mousepos, isMousePressed)
                boutonArreter.update(mousepos, isMousePressed)
                boutonCommencer.draw(affichage)
                boutonArreter.draw(affichage)
                
                if boutonCommencer.is_clicked(mousepos, isMousePressed):
                    waiting = False
                
                if boutonArreter.is_clicked(mousepos, isMousePressed):
                    pg.quit()
                    arret_episode = True
                    break
                
                pg.display.flip()
                clock.tick(60)

        if arret_episode:
            break
        
        # ! ------ Début de l'épisode ------
        episode += 1
        cpt = 0
        temps = 0
        done = False
        start_time = time.time()
        
        while not done:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
            
            affichage.fill((0, 0, 0))
            pg.draw.rect(affichage, (20, 20, 20), pg.Rect(width//2, 0, width//2, height))
            
            boutonArreter.update(pg.mouse.get_pos(), pg.mouse.get_pressed()[0])
            boutonArreter.draw(affichage)

            if boutonArreter.is_clicked(pg.mouse.get_pos(), pg.mouse.get_pressed()[0]):
                isManuel = True
                arretAuto = True
                break


            # Image avant l'action
            state_old = agent.get_state(env)
            
            # Affiche la frame actuelle
            frame = env.get_frame()
            hframe, wframe = frame.shape
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                imgpg = pg.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                imgmoit = pg.surfarray.make_surface(frame_rgb[debutimg:, :].swapaxes(0, 1))
                imgpg = pg.transform.scale(imgpg, (320, 240))
                imgmoit = pg.transform.scale(imgmoit, (320, 120))
                wframe, hframe = imgpg.get_size()

                affiche_texte("Flux vidéo:", (255, 255, 255), width//4, height//2 - hframe - 48, 20, True)
                affichage.blit(imgpg, (width//4 - wframe//2, height//2 - hframe - 24))

                affiche_texte("Flux traité:", (255, 255, 255), width//4, height//2 + hframe//2 - 24, 20, True)
                affichage.blit(imgmoit, (width//4 - wframe//2, height//2 + hframe//2))
                
            
            # Choix de l'action
            # Choix de l'action
            action = agent.get_action(state_old)

            # Envoi de l'action
            now = time.time()
            action_deja_comptabilisee = False # Empêche cpt d'augmenter en boucle à chaque micro-step
            
            while now - start_time < DELAI_ACTIONS:
                # Si l'action a déjà été pénalisée, on envoie un compteur virtuel à 0
                # pour éviter que env.step n'ajoute +1 en boucle pendant les 0.4s
                cpt_actuel = 0 if action_deja_comptabilisee else cpt
                
                # Exécution du step (comme dans ton code d'origine)
                state_new, reward, done, cpt_recu = env.step(action, cpt_actuel)
                
                # CORRECTION CNN : Redimensionnement immédiat à chaque réception d'image
                state_new = state_new.reshape(1, agent.input_height, agent.input_width)
                
                # Si le capteur détecte une déviation pour la première fois pendant cette action
                if cpt_recu > cpt_actuel and not action_deja_comptabilisee:
                    cpt += 1
                    action_deja_comptabilisee = True
                
                # Si done est True (ex: b'11' détecté), on coupe la boucle instantanément
                if done:
                    break
                
                now = time.time()
                
            start_time = time.time()
            # Train short memory
            agent.train_short_memory(state_old, action, reward, state_new, done)

            # Remember
            agent.remember(state_old, action, reward, state_new, done)

            print(f"cpt = {cpt}, action = {action}, reward = {reward}")
            
            # Affichage des infos
            affiche_texte(f"Épisode: {episode}", (255, 255, 255), width//2 + 20, height//2 - hframe//2, 20)
            affiche_texte(f"Nb d'Actions: {temps}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 22, 20)
            affiche_texte(f"Action: {action}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 44, 20)
            affiche_texte(f"Record: {record}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 66, 20)
            affiche_texte(f"Epsilon: {agent.epsilon:.4f}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 88, 20)
            
            temps += 1

            pg.display.flip()
            clock.tick(60)

        env.reset()
        time.sleep(DELAI_ACTIONS)

        # Retour en arrière
        actions_inverse = agent.get_memory(NB_RETOURS, temps)
        actions_inverse.append(3) # stop
        for action in actions_inverse:
            print(f"Action inverse: {action}")
            env.step(action, cpt, True)
            time.sleep(DELAI_ACTIONS+0.05)
            
        _, sensor = env.reset()
        print(sensor)
        if not arretAuto:
            isManuel = isManuelChoisi # Retour à automatique ou manuel selon le choix initial si il n'y a pas de problème de replacement

        agent.train_long_memory()
        agent.steps_done += 1
        
        if temps > record:
            record = temps
            agent.model.save(f"model_dqn{numModel}.pth")            
        
        print(f"Fin de l'épisode:\nÉpisode: {episode}, Nombre d'actions: {temps}, Record: {record}, Epsilon: {agent.epsilon}")
        temps_episodes.append(temps)
        temps_total += temps
        temps_episodes_moyen.append(temps_total / len(temps_episodes))

        with open(f"./resultats/{numModel}.csv", "w", newline='') as f:
            file = csv.writer(f)
            file.writerow([episode, temps_episodes[episode-1], temps_episodes_moyen[episode-1], record])


except Exception as e:
    print("ERREUR!!!!!: ", e)
    env.reset()
    titre = input("Titre: ")
    if titre is None or titre == "":
        nb = len(os.listdir("./resultats"))
        titre = f"resultats{nb}"
    titre = os.path.join("./resultats", f"{titre}.csv")
    

    with open(titre, "w", newline='') as f:
        file = csv.writer(f)
        file.writerow(["Episode", "Nombre d'actions réalisées", "Nombre d'action moyen", "Record"])
        for i in range(len(temps_episodes)):
            file.writerow([i, temps_episodes[i], temps_episodes_moyen[i], record])
    
    with open(os.path.join("./resultats", f"{numModel}.txt"), "w+") as f:
        f.write(f"Modèle: model_dqn{numModel}.pth\n")
        f.write(f"Gamma: {agent.gamma}\n")
        f.write(f"LR: {agent.trainer.lr}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Neurones couche 1: 512\n")
        f.write(f"Neurones couche 2: 256\n")
        f.write(f"EPS_DECAY: {EPS_DECAY}\n")
        f.write(f"DELAI_ACTIONS: {DELAI_ACTIONS}\n")
        f.write(f"NB_RETOURS: {NB_RETOURS}\n")
        f.write(f"Steps done: {agent.steps_done}\n")
        f.write(f"CSV: {titre[11:]}\n")

        
finally:
    env.reset()
    titre = input("Titre: ")
    if titre is None or titre == "":
        nb = len(os.listdir("./resultats"))
        titre = f"resultats{nb}"
    titre = os.path.join("./resultats", f"{titre}.csv")
    

    with open(titre, "w", newline='') as f:
        file = csv.writer(f)
        file.writerow(["Episode", "Nombre d'actions réalisées", "Nombre d'action moyen", "Record"])
        for i in range(len(temps_episodes)):
            file.writerow([i, temps_episodes[i], temps_episodes_moyen[i], record])
    
    with open(os.path.join("./resultats", f"{numModel}.txt"), "w+") as f:
        f.write(f"Modèle: model_dqn{numModel}.pth\n")
        f.write(f"Gamma: {agent.gamma}\n")
        f.write(f"LR: {agent.trainer.lr}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Neurones couche 1: 512\n")
        f.write(f"Neurones couche 2: 256\n")
        f.write(f"EPS_DECAY: {EPS_DECAY}\n")
        f.write(f"DELAI_ACTIONS: {DELAI_ACTIONS}\n")
        f.write(f"NB_RETOURS: {NB_RETOURS}\n")
        f.write(f"Steps done: {agent.steps_done}\n")
        f.write(f"CSV: {titre[11:]}\n")
        

    pg.quit()
    cv2.destroyAllWindows()