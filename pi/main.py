import socket
import struct
import time
import cv2
from picamera2 import Picamera2
from libcamera import Transform
import controle as c
import select

# SERVER_IP = "172.20.10.5"
SERVER_IP = "192.168.1.136"
SERVER_PORT = 9998
TARGET_FPS = 15.0 

# --- Setup socket ---
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, SERVER_PORT))
client_socket.settimeout(0.05)  # non-blocking recv (50ms timeout)
print("Connected to server.")

# --- Setup camera ---
picam2 = Picamera2()
picam2.rotate = 180
picam2.configure(picam2.create_video_configuration(main={"size": (320, 240)}))
# picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
Transform(vflip=1)
picam2.start()
time.sleep(2)
print(f"Camera started (grayscale mode, {TARGET_FPS:.1f} FPS limit)...")

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
frame_interval = 1.0 / TARGET_FPS
last_capture = 0.0

try:
    while True:
        # Limit FPS
        now = time.time()
        if now - last_capture < frame_interval:
            time.sleep(frame_interval - (now - last_capture))
        last_capture = time.time()

        # Capture frame
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Encode as JPEG
        result, img_encode = cv2.imencode('.jpg', frame, encode_param)
        data = img_encode.tobytes()

        # Send frame
        client_socket.sendall(struct.pack('<L', len(data)))
        client_socket.sendall(data)

        # Try receiving command (non-blocking)
        ready, _, _ = select.select([client_socket], [], [], 0)
        try: 
            if ready:
                cmd = client_socket.recv(1)
                if not cmd:
                    break
                d = cmd.decode()
                print("Received:", d)
                if d == 'F': c.forward()
                elif d == 'B': c.backward()
                elif d == 'L': c.left()
                elif d == 'R': c.right()
                elif d == 'S': c.stop(); c.center()
                elif d == 'C': c.center()
        except socket.timeout:
            # No command received; continue streaming
            pass
        except (ConnectionResetError, OSError) as e:
            print("Connection lost:", e)
            break

finally:
    client_socket.close()
    picam2.close()
    c.stop()
    c.center()
    c.GPIO.cleanup()
    print("Client shutdown cleanly.")
