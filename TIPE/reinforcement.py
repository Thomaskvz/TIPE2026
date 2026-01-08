import socket
import struct
import cv2
import numpy as np
import sys
import time
import pygame as pg
import environment
import random
import model_dqn
import replay_buffer

HOST = '' if len(sys.argv) < 2 else sys.argv[1]
PORT = 9991 if len(sys.argv) < 3 else int(sys.argv[2])

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))

server_socket.listen(0)
print(f"Ecoute {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print("Connecté par", addr)

connection = conn.makefile('rb')
prev_time = time.time()


# Initialisation de l'environnement, du modèle et du buffer de replay

env = environment.Environment(conn, connection)
model = model_dqn.build_dqn()
target = model_dqn.build_dqn()
target.set_weights(model.get_weights())
buffer = replay_buffer.ReplayBuffer(size=100000)

# Initialisation de l'affichage

pg.init()
pg.display.set_caption("Le meilleur TIPE de l'UNIVERS")
height = 240
width = 640
affichage = pg.display.set_mode((width,height))

text_font = pg.font.SysFont("Helvetica", 18)

def affiche_texte(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    affichage.blit(img, (x,y))

action_map={
    0:b'F',
    1:b'R',
    2:b'L'
}

affichage.fill((0,0,0))
affiche_texte("En attente du démarrage du training RL...", text_font, (255, 255, 255), 20, height//2)

epsilon = 1.0  # Probabilité d'exploration

episodes = 100
for episode in range(1, episodes+1):
    frame, sensor = env.reset() # Les états sont les images envoyées par la Pi
    hframe, wframe = frame.shape
    done = False
    score = 0 
    count=0
    input("Appuyez sur Entrée pour démarrer l'épisode...")
    print("Starting episode ", episode)


    while not done:
        img = frame/255.0  # Normalisation de l'image

        if random.random() < epsilon:
            action = random.randint(0,2)  # Exploration: action aléatoire
        else:
            action = np.argmax(model.predict(img[hframe//2:,:].flatten().reshape(1, -1)))  # Exploitation: action selon le modèle

        nframe, sensor, reward, done, count = env.step(action, count)
        buffer.add(frame, action, reward, nframe, done)
        frame = nframe
        score+=reward


        #TODO à VERIFIER
        if len(buffer)>32:
            states, actions, rewards, next_states, dones = buffer.sample(32)
            states = states[:, hframe//2:, :].reshape(32, -1)/255.0
            next_states = next_states[:, hframe//2:, :].reshape(32, -1)/255.0

            q_values = model.predict(states)
            next_q_values = model.predict(next_states)

            for i in range(32):
                target = rewards[i]
                if not dones[i]:
                    target += 0.99 * np.amax(next_q_values[i])
                q_values[i][actions[i]] = target

            model.train_on_batch(states, q_values)

        # ? Affichage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        imgpg = pg.surfarray.make_surface(frame.swapaxes(0, 1))
        affichage.fill((0,0,0))
        affichage.blit(imgpg,(0,0))
        hframe, wframe = frame.shape
        affiche_texte(f"Mode: Reinforcement Learning", text_font, (255, 255, 255), wframe+2,2)
        affiche_texte(f"Action: {action_map[action]}", text_font, (255,255,255), wframe+2, 20)
        affiche_texte(f"Capteur: {sensor}", text_font, (255,255,255), wframe+2, 38)
        pg.display.flip() # Met à jour la fenêtre pygame

    epsilon = max(0.01, epsilon * 0.995)  # Décroissance de epsilon
    target.set_weights(model.get_weights())  # Mise à jour du modèle cible
    model.save("model_dqn.h5")
    print(f'Episode:{episode} Score:{score}')
    


