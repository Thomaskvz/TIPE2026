# run_rl.py
import numpy as np
import tensorflow as tf
from car_env import CarEnv


def run(conn, connection):
    model = tf.keras.models.load_model("models/car_dqn.h5", custom_objects={"mse": tf.keras.metrics.MeanSquaredError})
    env = CarEnv(conn, connection)

    state = env.reset()

    while True:
        action = np.argmax(model.predict(state[None], verbose=0))
        state, _, done = env.step(action)
        if done:
            state = env.reset()
