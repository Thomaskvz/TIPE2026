import struct
import cv2
import numpy as np
import socket


class Environment:
    def __init__(self):
        HOST = ''
        PORT = 9991

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((HOST, PORT))

        server_socket.listen(0)
        print(f"Ecoute {HOST}:{PORT}...")
        self.conn, addr = server_socket.accept()
        print("Connecté par", addr)

        self.connection = self.conn.makefile('rb')

        self.action_map={
            0:b'F',
            1:b'R',
            2:b'L'
        }


    def reset(self):
        self.conn.sendall(b"S")
        frame, sensor = self.process_image()
        return frame, sensor



    def step(self, action, cpt):
        self.conn.sendall(b'F')
        self.conn.sendall(self.action_map[action])
        frame, sensor = self.process_image()
        done=False
        reward=1   
                
        if sensor==b'01' or sensor==b'10':
            cpt+=1
        if cpt>=2:
            done=True
            reward=-10

        return frame, reward, done, cpt
    

    def process_image(self):
        self.conn.sendall(b'I')
        header = self.connection.read(struct.calcsize('<LL'))
        if not header or len(header) < struct.calcsize('<LL'):
            raise RuntimeError("Socket closed or corrupted header")
        image_len, capt_len = struct.unpack('<LL', header)
        if not image_len or not capt_len:
            return

        # * Lecture des données de l'image
        data = b''
        while len(data) < image_len:
            more = self.connection.read(image_len - len(data))
            if not more:
                break
            data += more

        # * Decode les bytes en image
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        frame = cv2.flip(frame,-1)

        if frame is None:
            raise RuntimeError("Image decode failed")

        sensor = b''
        while len(sensor) < capt_len:
            more = self.connection.read(capt_len - len(sensor))
            if not more:
                break
            sensor += more

        hframe, wframe = frame.shape
        img = frame[hframe//2:,:].flatten()/255

        return img, sensor