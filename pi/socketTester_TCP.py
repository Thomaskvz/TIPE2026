import socket
import struct
import time
import cv2
import select

# SERVER_IP = "172.20.10.5"
SERVER_IP = "192.168.1.121"
SERVER_PORT = 9991
TARGET_FPS = 15.0 

# --- Setup socket ---
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, SERVER_PORT))
client_socket.settimeout(0.05)  # non-blocking recv (50ms timeout)
print("Connected to server.")

# --- Setup camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(2)
print(f"Camera started (grayscale mode, {TARGET_FPS:.1f} FPS limit)...")

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
frame_interval = 1.0 / TARGET_FPS
last_capture = 0.0

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

        if d == 'F': print("Forward")
        elif d == 'B': print("Backward")
        elif d == 'L': print("Left")
        elif d == 'R': print("Right")
        elif d == 'S':
            print("Stop")
        elif d == 'C': print("Center")
        elif d == 'I':
            # ----------------------
            # Capture frame
            # ----------------------
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame, skipping")
                continue

            frame = cv2.resize(frame, (320, 240), interpolation =cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.flip(frame, 0)  # Vertical flip

            # Encode JPEG
            result, img_encode = cv2.imencode('.jpg', frame, encode_param)
            if not result:
                print("JPEG encoding failed, skipping frame")
                continue
            img_bytes = img_encode.tobytes()

            # Get sensor
            sensor = b"01"  # Placeholder for actual sensor data

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
    print("Client shutdown cleanly.")
