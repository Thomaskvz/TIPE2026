import torch
import random
import numpy as np
from collections import deque
from environment import Environment
from model import Linear_QNet, QTrainer
import time
import os
import csv
import pygame as pg
import cv2
import sys
from modules_pygame.boutons import Button

MAX_MEMORY = 100_000
BATCH_SIZE = 256
LR = 0.001
EPS_DECAY = 30
EPS_START = 0.9
DELAI_ACTIONS = 0.5

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

class Agent:
    def __init__(self):
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(120*320, 256, 3)
        self.trainer = QTrainer(self.model, LR, self.gamma)
        self.steps_done = 0

    def get_state(self, env):
        return env.process_image()[0]
    
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

    def get_action(self, state):
        self.epsilon = EPS_START * np.exp(-1. * self.steps_done / EPS_DECAY)
        if random.random() < self.epsilon:
            action = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            pred = self.model(state0)
            action = torch.argmax(pred).item()
            
        return action

temps_episodes = []
temps_episodes_moyen = []
temps_total = 0
record = 0
numModel = len(os.listdir("./models"))

agent = Agent()
env = Environment()

episode = 0

boutonCommencer = Button(width//2 - 100, height//2, 200, 80, "Commencer", pg.font.SysFont("Helvetica", 28), (0, 225, 0), (0, 0, 0), (255, 255, 255), (0, 200, 0))
boutonArreter = Button(width//2 - 100, height//2 + 100, 200, 80, "Arrêter", pg.font.SysFont("Helvetica", 28), (225, 0, 0), (0, 0, 0), (255, 255, 255), (200, 0, 0))

try:
    while True:
        # ! ------ Écran d'attente avant l'épisode ------
        waiting = True
        while waiting:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
            
            mousepos, isMousePressed = pg.mouse.get_pos(), pg.mouse.get_pressed()[0]
            affichage.fill((0, 0, 0))
            
            # Affiche la dernière frame de l'env
            frame = env.get_frame()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                imgpg = pg.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                hframe, wframe = frame.shape
                affichage.blit(imgpg, (width//4 - wframe//2, height//2 - hframe//2))
            
            affiche_texte("En attente du prochain épisode...", (255, 255, 255), width//2, height//2 - 150, 28, True)
            affiche_texte(f"Épisode: {episode}", (255, 255, 255), width//2, height//2 - 80, 24, True)
            affiche_texte(f"Record: {record}", (255, 255, 255), width//2, height//2 - 40, 24, True)
            
            boutonCommencer.update(mousepos, isMousePressed)
            boutonArreter.update(mousepos, isMousePressed)
            boutonCommencer.draw(affichage)
            boutonArreter.draw(affichage)
            
            if boutonCommencer.is_clicked(mousepos, isMousePressed):
                waiting = False
            
            if boutonArreter.is_clicked(mousepos, isMousePressed):
                pg.quit()
                raise KeyboardInterrupt("Training stopped by user")
            
            pg.display.flip()
            clock.tick(60)

        
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
            
            # Image avant l'action
            state_old = agent.get_state(env)
            
            # Affiche la frame actuelle
            frame = env.get_frame()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                imgpg = pg.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                hframe, wframe = frame.shape
                
                pg.draw.rect(affichage, (20, 20, 20), pg.Rect(width//2, 0, width//2, height))
                affiche_texte("Flux vidéo:", (255, 255, 255), width//4, height//2 - hframe//2 - 24, 20, True)
                affichage.blit(imgpg, (width//4 - wframe//2, height//2 - hframe//2))
            
            # Choix de l'action
            action = agent.get_action(state_old)

            # Envoi de l'action
            state_new, reward, done, cpt = env.step(action, cpt)

            # Train short memory
            agent.train_short_memory(state_old, action, reward, state_new, done)

            # Remember
            agent.remember(state_old, action, reward, state_new, done)

            print(f"cpt = {cpt}, action = {action}, reward = {reward}")
            
            # Affichage des infos
            affiche_texte(f"Épisode: {episode}", (255, 255, 255), width//2 + 20, height//2 - hframe//2, 20)
            affiche_texte(f"Actions: {temps}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 22, 20)
            affiche_texte(f"Action: {action}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 44, 20)
            affiche_texte(f"Record: {record}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 66, 20)
            affiche_texte(f"Epsilon: {agent.epsilon:.4f}", (255, 255, 255), width//2 + 20, height//2 - hframe//2 + 88, 20)

            now = time.time()
            while now - start_time < DELAI_ACTIONS:
                _, _, done, cpt = env.step(action, cpt)
                if done:
                    break
                now = time.time()
            start_time = time.time()
            temps += 1

            pg.display.flip()
            clock.tick(60)

        env.reset()
        agent.train_long_memory()
        agent.steps_done += 1
        
        if temps > record:
            record = temps
            agent.model.save(f"model_dqn{numModel}.pth")
        
        print(f"Fin de l'épisode:\nÉpisode: {episode}, Nombre d'actions: {temps}, Record: {record}, Epsilon: {agent.epsilon}")
        temps_episodes.append(temps)
        temps_total += temps
        temps_episodes_moyen.append(temps_total / len(temps_episodes))
        
finally:
    titre = input("Titre: ")
    if titre is None or titre == "":
        nb = len(os.listdir("./resultats"))
        titre = f"resultats{nb}.csv"

    with open(titre, "w") as f:
        file = csv.writer(f)
        file.writerow(["Episode", "Nombre d'actions réalisées", "Nombre d'action moyen", "Record"])
        for i in range(len(temps_episodes)):
            file.writerow([i, temps_episodes[i], temps_episodes_moyen[i], record])
    
    pg.quit()
    cv2.destroyAllWindows()