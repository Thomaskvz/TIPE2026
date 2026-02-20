import socket
import struct
import time
import cv2
from picamera2 import Picamera2
from libcamera import Transform
import controle as c
import capteur as cap

SERVER_IP = "172.20.10.5"  # IP du PC
SERVER_PORT = 9991 # Port d'envoi d'image
PI_LISTEN_PORT = 9992  # Port de réception de commande
TARGET_FPS = 30.0 

# --- Setup UDP socket pour envoi d'image ---
send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print("UDP send socket created.")

# --- Setup UDP socket pour réception de commande ---
recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_socket.bind(('', PI_LISTEN_PORT))
recv_socket.settimeout(0.05)
print(f"UDP listen socket bound to port {PI_LISTEN_PORT}")


# Setup camera
picam2 = Picamera2()
picam2.rotate = 180
picam2.configure(picam2.create_video_configuration(main={"size": (320, 240)}))
Transform(vflip=1)
picam2.start()
time.sleep(2)
print("Camera started.")

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
frame_interval = 1.0 / TARGET_FPS
last_capture = 0.0

try:
    while True:
        # ----------------------
        # Essaye de recevoir une commande (non bloquant)
        # ----------------------
        try:
            cmd, addr = recv_socket.recvfrom(1)
            if not cmd:
                continue
            
            d = cmd.decode()
            
            if d == 'F': c.forward()
            elif d == 'B': c.backward()
            elif d == 'L': c.left()
            elif d == 'R': c.right()
            elif d == 'S': c.stop()
            elif d == 'C': c.center()
            else:
                print("Commande inconnue:", d)
        
        except socket.timeout:
            pass
        
        # ----------------------
        # Envoi d'image
        # ----------------------
        now = time.time()
        if now - last_capture >= frame_interval:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Encode en JPEG
            result, img_encode = cv2.imencode('.jpg', frame, encode_param)
            if not result:
                print("JPEG encoding failed, skipping frame")
                continue
            img_bytes = img_encode.tobytes()

            # Capteur
            sensor = cap.detection()
            print("Capteur:", sensor)

            # ----------------------
            # Envoie header + image + sensor
            # ----------------------
            header = struct.pack('<LL', len(img_bytes), len(sensor))
            payload = header + img_bytes + sensor
            
            try:
                send_socket.sendto(payload, (SERVER_IP, SERVER_PORT))
            except OSError as e:
                print("Envoi échoué:", e)
            
            last_capture = now

except KeyboardInterrupt:
    print("Keyboard interrupt")

except (ConnectionResetError, OSError) as e:
    print("Connection perdue:", e)

finally:
    picam2.close()
    c.stop()
    c.center()
    c.GPIO.cleanup()
    send_socket.close()
    recv_socket.close()
    print("Arrêt du programme.")
