# train_rl.py
import numpy as np
from car_env import CarEnv
from dqn_model import build_dqn
from replay_buffer import ReplayBuffer

def train(conn, connection):
    env = CarEnv(conn, connection)  # reuse socket from main.py
    model = build_dqn()
    target = build_dqn()
    target.set_weights(model.get_weights())

    buffer = ReplayBuffer()
    gamma = 0.99
    epsilon = 1.0

    for episode in range(1000):
        conn.sendall(b'S')  # stop
        input("Press Enter to start the next episode...")
        state = env.reset()
        total_reward = 0
        count = 0
        print("Starting episode", episode)
        while True:
            if np.random.rand() < epsilon:
                action = np.random.randint(3)
            else:
                action = np.argmax(model.predict(state[None], verbose=0))
            
            next_state, reward, done, count = env.step(action, count)
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) > 32:
                s, a, r, s2, d = buffer.sample(32)
                q_next = target.predict(s2, verbose=0)
                q = model.predict(s, verbose=0)

                for i in range(32):
                    q[i, a[i]] = r[i] + gamma * np.max(q_next[i]) * (1 - d[i])

                model.train_on_batch(s, q)

            if done:
                break

        epsilon *= 0.995
        target.set_weights(model.get_weights())
        model.save("models/car_dqn.h5")

        print("Episode", episode, "Reward", total_reward)
