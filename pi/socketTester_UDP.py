import socket
import struct
import time
import cv2
import random

SERVER_IP = "192.168.1.121"  # PC's IP
SERVER_PORT = 9991
PI_LISTEN_PORT = 9992  # Pi listens for commands on this port
TARGET_FPS = 30.0 

# --- Setup UDP socket for sending frames ---
send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print("UDP send socket created.")

# --- Setup UDP socket for receiving commands ---
recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_socket.bind(('', PI_LISTEN_PORT))
recv_socket.settimeout(0.05)
print(f"UDP listen socket bound to port {PI_LISTEN_PORT}")

# --- Setup camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(2)
print(f"Camera started (grayscale mode, {TARGET_FPS:.1f} FPS limit)...")

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
frame_interval = 1.0 / TARGET_FPS
last_capture = 0.0

cpt = 0

try:
    while True:
        # ----------------------
        # Try to receive command (non-blocking)
        # ----------------------
        try:
            cmd, addr = recv_socket.recvfrom(1)
            if not cmd:
                continue
            
            d = cmd.decode()
            
            if d == 'F': print("Forward")
            elif d == 'B': print("Backward")
            elif d == 'L': print("Left")
            elif d == 'R': print("Right")
            elif d == 'S': print("Stop")
            elif d == 'C': print("Center")
            else:
                print("Unknown command:", d)
        
        except socket.timeout:
            pass
        
        # ----------------------
        # Send frame periodically
        # ----------------------
        now = time.time()
        if now - last_capture >= frame_interval:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame, skipping")
                last_capture = now
                continue

            frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.flip(frame, 0)

            result, img_encode = cv2.imencode('.jpg', frame, encode_param)
            if not result:
                print("JPEG encoding failed, skipping frame")
                last_capture = now
                continue
            img_bytes = img_encode.tobytes()

            sensor = b'00'  # Placeholder for actual sensor data
            cpt = (cpt+1)%20
            if cpt==0:
                sensor = b'01'

            header = struct.pack('<LL', len(img_bytes), len(sensor))
            payload = header + img_bytes + sensor
            
            try:
                send_socket.sendto(payload, (SERVER_IP, SERVER_PORT))
            except OSError as e:
                print("Failed to send frame:", e)
            
            last_capture = now

except KeyboardInterrupt:
    print("Interrupted by user.")

except Exception as e:
    print("Error:", e)

finally:
    send_socket.close()
    recv_socket.close()
    cap.release()
    print("Client shutdown cleanly.")