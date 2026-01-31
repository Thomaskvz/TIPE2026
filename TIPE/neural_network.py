import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import namedtuple, deque
from torch import optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from itertools import count
import matplotlib
import math
from environment import Environment as env


device= torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Utilisation du GPU/CPU pour l'entraînement
print("device:", device)

Transition= namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN,self).__init__()
        self.layer1 = nn.Linear(n_observations,128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128,n_actions)

    def forward(self,x):
        x= F.relu(self.layer1(x))
        x= F.relu(self.layer2(x))
        return self.layer3(x)


BATCH_SIZE = 128 # Taille du batch pour l'entraînement
GAMMA = 0.99 # Facteur de réduction des récompenses futures
EPS_START = 0.9 # Valeur de départ pour epsilon dans la stratégie epsilon-greedy
EPS_END = 0.001 # Valeur finale pour epsilon
EPS_DECAY = 2500 # Vitesse de décroissance de epsilon
TAU = 0.005 # Fréquence de mise à jour du réseau cible
LR = 3e-4 # Taux d'apprentissage

n_actions = 3
n_observations = 120*320 
state, sensor = env.reset()

policy_net = DQN(n_observations,n_actions).to(device)
target_net = DQN(n_observations,n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = [] #pour le tracé des durées des épisodes

def optimize_model():
    if len(memory)<BATCH_SIZE:
        return 
    transistions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transistions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) #concatène les états

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device) # TODO : comprendre
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values.squeeze(), expected_state_action_values)   
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(),100)
    optimizer.step()


if torch.backends.mps.is_available():
    num_episodes= 500
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    state, sensor = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    cpt=0
    for t in count():
        action = select_action(state)
        observation, reward, done, cpt = env.step(action.item(),cpt)
        reward = torch.tensor([reward], device=device)
        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()

        target_net_state_dict = target_net.state_dict() #TODO: comprende
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            break

print('Entraînement terminé')
torch.save(policy_net.state_dict(), os.path.join("TIPE","neural_network.pth"))





