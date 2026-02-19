import torch
import random
import numpy as np
from collections import deque
from environment import Environment
from model import Linear_QNet, QTrainer
import time
import os
import csv

MAX_MEMORY = 100_000 # Nombre de valeurs conservées dans le buffer
BATCH_SIZE = 256
LR = 0.001 # Learning Rate
EPS_DECAY = 30 # Il s'agit du \tau dans e^-t/\tau
EPS_START = 0.9
NB_EPISODES = 100
DELAI_ACTIONS = 0.5

class Agent:
    def __init__(self):
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY) # File contenant les anciens résultats
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

    def get_action(self,state):
        self.epsilon = EPS_START* np.exp(-1. * self.steps_done / EPS_DECAY)
        if random.random() < self.epsilon:
            action = random.randint(0,2)
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

for episode in range(NB_EPISODES):
    input("Appuyez sur Entrée pour commencer l'épisode...")
    cpt = 0
    temps = 0
    done = False
    start_time = time.time()
    while not done:
        print("cpt = ", cpt)
        temps += 1
        # Image avant l'action
        state_old = agent.get_state(env)

        # Choix de l'action, ou bien aléatoire, ou bien choisie par IA
        action = agent.get_action(state_old)

        print(f"Action choisie: {action}")
        # Envoi de l'action
        state_new, reward, done, cpt = env.step(action, cpt)

        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # remember
        agent.remember(state_old, action, reward, state_new, done)

        now = time.time()
        while now - start_time < DELAI_ACTIONS:
            _, _, done, cpt = env.step(action,cpt)
            if done:
                break
            now = time.time()
        start_time = time.time()

    env.reset()
    agent.train_long_memory()
    agent.steps_done += 1
    if temps > record:
        record = temps
        agent.model.save(f"model_dqn{numModel}.pth")
    print(f"Fin de l'épisode:\nEpisode: {episode}, Nombre d'actions: {temps}, Record: {record}, Epsilon: {agent.epsilon}")
    temps_episodes.append(temps)
    temps_total += temps
    temps_episodes_moyen.append(temps_total/len(temps_episodes))

titre = input("Titre: ")
if titre is None or titre == "":
    nb = len(os.listdir("./resultats"))
    titre = f"resultats{nb}.csv"

with open(titre, "w") as f:
    file = csv.writer(f)
    file.writerow(["Episode", "Nombre d'actions réalisées", "Nombre d'action moyen", "Record"])
    for i in range(len(temps_episodes)):
        file.writerow([i, temps_episodes[i], temps_episodes_moyen[i], record])
