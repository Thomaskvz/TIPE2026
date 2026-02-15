import torch
from model import Linear_QNet
from environment import Environment
import os
import time
import cv2

DELAI_ACTIONS = 0.5

env = Environment()
nnmodel = Linear_QNet(120*320, 256, 3)

numModel = input("Choisissez le numéro du model: ")
pathmodel = os.path.join("./models", f"model_dqn{numModel}.pth")
nnmodel.load_state_dict(torch.load(pathmodel, weights_only=True))

start_time = time.time()

while True:
    state, _ = env.process_image()

    cv2.imshow("Image", state)

    state0 = torch.tensor(state, dtype=torch.float)
    pred = nnmodel(state0)
    action = torch.argmax(pred).item()
    
    env.step(action, 0)

    now = time.time()
    if now - start_time < DELAI_ACTIONS:
        time.sleep(DELAI_ACTIONS - (now - start_time))
    start_time = time.time()
    