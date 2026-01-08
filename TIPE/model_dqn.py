import TensorFlow as tf
from tensorflow.keras import layers

def build_dqn():
    model = tf.keras.Sequential([
        layers.Input(shape=(320, 120, 1)),
        layers.Conv2D(32, (8, 8), strides=4, activation='relu'),
        layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
        layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(3, activation='linear')  # 3 actions: forward, right, left
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

