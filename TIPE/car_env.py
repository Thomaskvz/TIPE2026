# car_env.py
import cv2
import numpy as np
import struct

class CarEnv:
    def __init__(self, conn, connection):
        self.conn = conn
        self.connection = connection
        self.action_map = {
            0: b'L',
            1: b'F',
            2: b'R'
        }
        self.conn.settimeout(5.0)

    def reset(self):
        self.conn.sendall(b'S')
        try:
            state, _ = self._recv()
        except Exception as e:
            print("Reset failed:", e)
            state = np.zeros((84, 84, 1))  # default frame
        return self._preprocess(state)

    def step(self, action, count):
        self.conn.sendall(self.action_map[action])

        frame, sensors = self._recv()
        done = False
        if sensors != b'00':
            count += 1
        if count >=1:
            done = True

        reward = 0.1
        if done:
            reward = -10.0

        return self._preprocess(frame), reward, done, count

    def _recv(self):
        try:
            header = self.connection.read(struct.calcsize('<LL'))
            if len(header) < 8:
                raise TimeoutError("Header incomplete")
            image_len, capt_len = struct.unpack('<LL', header)

            img_bytes = self.connection.read(image_len)
            if len(img_bytes) < image_len:
                raise TimeoutError("Image incomplete")
            frame = cv2.imdecode(
                np.frombuffer(img_bytes, np.uint8),
                cv2.IMREAD_GRAYSCALE
            )

            sensors = self.connection.read(capt_len)
            if len(sensors) < capt_len:
                raise TimeoutError("Sensors incomplete")
        except TimeoutError:
            print("Warning: _recv() timed out. Returning default values.")
            frame = np.zeros((240, 320), dtype=np.uint8)
            sensors = b'LC'
        return frame, sensors

    def _preprocess(self, frame):
        frame = cv2.resize(frame, (84, 84))
        frame = frame / 255.0
        return frame[..., None]
