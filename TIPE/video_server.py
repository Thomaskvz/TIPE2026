import socket
import struct
import cv2
import numpy as np
import sys
import time
import pygame as pg
import ia_image


HOST = '' if len(sys.argv) < 2 else sys.argv[1]
PORT = 9998 if len(sys.argv) < 3 else int(sys.argv[2])

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(0)
print(f"Listening on {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print("Connected by", addr)

connection = conn.makefile('rb')
prev_time = time.time()

pg.init()
pg.display.set_mode((250,250))

i = 0

test = False # A ENLEVER

try:
    while True:
        image_len_data = connection.read(struct.calcsize('<L'))
        if not image_len_data:
            break
        image_len = struct.unpack('<L', image_len_data)[0]
        if not image_len:
            break

        # Read frame bytes
        data = b''
        while len(data) < image_len:
            more = connection.read(image_len - len(data))
            if not more:
                break
            data += more

        # Decode grayscale image
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        frame = cv2.flip(frame,-1)

        # Compute FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time+0.000001)
        prev_time = curr_time

        # # Show image with FPS overlay
        # cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)
        cv2.imshow('Raspberry Pi Stream (B/W)', frame)

        
        img = frame/255
        pred = ia_image.clf.predict(img.flatten().reshape(1,-1))
        print(ia_image.definition[pred[0]])

        if test == False:
            test = True
            print(frame)

        """for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP:
                    cv2.imwrite(f"training_data/avance/{i}.jpg", frame)
                if event.key == pg.K_RIGHT:
                    cv2.imwrite(f"training_data/droite/{i}.jpg", frame)
                if event.key == pg.K_LEFT:
                    cv2.imwrite(f"training_data/gauche/{i}.jpg", frame)
                if event.key == pg.K_DOWN:
                    cv2.imwrite(f"training_data/recule/{i}.jpg", frame)
                if event.key == pg.K_t:
                    cv2.imwrite(f"test/img_test.jpg", frame)
                i+=1
        


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break"""
    
        for event in pg.event.get():

            if event.type == pg.QUIT:
                sys.exit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_z:      # forward
                    conn.sendall(b'F')
                    print("Forward")
                elif event.key == pg.K_s:    # backward
                    conn.sendall(b'B')
                    print("Backward")
                elif event.key == pg.K_q:    # left
                    conn.sendall(b'L')
                    print("Left")
                elif event.key == pg.K_d:    # right
                    conn.sendall(b'R')
                    print("Right")
                elif event.key == pg.K_SPACE:
                    conn.sendall(b'S')
                    print("Stop")

            elif event.type == pg.KEYUP:
                # stop the car when key is released
                conn.sendall(b'S')






finally:
    cv2.destroyAllWindows()
    connection.close()
    conn.close()
    server_socket.close()
