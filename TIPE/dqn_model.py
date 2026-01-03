# dqn_model.py
import tensorflow as tf
from tensorflow.keras import layers

def build_dqn():
    model = tf.keras.Sequential([
        layers.Input(shape=(84, 84, 1)),
        layers.Conv2D(32, 8, strides=4, activation='relu'),
        layers.Conv2D(64, 4, strides=2, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(3)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='mse'
    )
    return model
