import socket
import struct
import time
import cv2
from picamera2 import Picamera2
from libcamera import Transform
import controle as c
import capteur as cap

SERVER_IP = "172.20.10.5"
SERVER_PORT = 9991

# Setup socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, SERVER_PORT))
client_socket.settimeout(None)  # BLOCKING mode
print("Connected to server.")

# Setup camera
picam2 = Picamera2()
picam2.rotate = 180
picam2.configure(picam2.create_video_configuration(main={"size": (320, 240)}))
Transform(vflip=1)
picam2.start()
time.sleep(2)
print("Camera started.")

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

try:
    while True:
        # ----------------------
        # Wait for a command
        # ----------------------
        cmd = client_socket.recv(1)  # BLOCKING
        if not cmd:
            print("Server disconnected.")
            break

        d = cmd.decode()
        #print("Received command:", d)

        if d == 'F': c.forward()
        elif d == 'B': c.backward()
        elif d == 'L': c.left(); c.forward()
        elif d == 'R': c.right(); c.forward()
        elif d == 'S':
            c.stop()
            c.center()
        elif d == 'C': c.center()
        elif d == 'I':
            # ----------------------
            # Capture frame
            # ----------------------
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Encode JPEG
            result, img_encode = cv2.imencode('.jpg', frame, encode_param)
            if not result:
                print("JPEG encoding failed, skipping frame")
                continue
            img_bytes = img_encode.tobytes()

            # Get sensor
            sensor = cap.detection()
            print("Sensor data:", sensor)

            # ----------------------
            # Send header + image + sensor
            # ----------------------
            header = struct.pack('<LL', len(img_bytes), len(sensor))
            client_socket.sendall(header)
            client_socket.sendall(img_bytes)
            client_socket.sendall(sensor)


        else:
            print("Unknown command:", d)

except KeyboardInterrupt:
    print("Interrupted by user.")

except (ConnectionResetError, OSError) as e:
    print("Connection lost:", e)

finally:
    client_socket.close()
    picam2.close()
    c.stop()
    c.center()
    c.GPIO.cleanup()
    print("Client shutdown cleanly.")
