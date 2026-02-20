import struct
import cv2
import numpy as np
import socket


class Environment:
    def __init__(self):
        PI_IP = "192.168.1.121"
        PI_SEND_PORT = 9991  # Port where Pi sends frames
        PI_CMD_PORT = 9992   # Port where Pi listens for commands

        # UDP socket for receiving frames from Pi
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(('', PI_SEND_PORT))
        self.udp_socket.settimeout(0.1)
        
        self.pi_ip = PI_IP
        self.pi_cmd_port = PI_CMD_PORT
        
        print(f"Listening on port {PI_SEND_PORT} for UDP frames from Pi")

        self.action_map = {
            0: b'F',
            1: b'R',
            2: b'L'
        }
        
        self.last_frame = None
        self.last_sensor = b''


    def reset(self):
        self.udp_socket.sendto(b'S', (self.pi_ip, self.pi_cmd_port))
        frame, sensor = self.process_image()
        return frame, sensor



    def step(self, action, cpt):
        self.udp_socket.sendto(self.action_map[action], (self.pi_ip, self.pi_cmd_port))
        frame, sensor = self.process_image()
        done=False
        reward=1   
                
        if sensor==b'01' or sensor==b'10':
            cpt+=1
        if cpt>=2:
            done=True
            reward=-10
            self.udp_socket.sendto(b'S', (self.pi_ip, self.pi_cmd_port))

        return frame, reward, done, cpt
    

    def process_image(self):
        try:
            data, addr = self.udp_socket.recvfrom(65535)
            
            if len(data) >= struct.calcsize('<LL'):
                header_size = struct.calcsize('<LL')
                image_len, capt_len = struct.unpack('<LL', data[:header_size])
                
                if len(data) >= header_size + image_len + capt_len:
                    img_bytes = data[header_size:header_size + image_len]
                    sensor = data[header_size + image_len:header_size + image_len + capt_len]
                    
                    # Decode image
                    frame = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if frame is None:
                        raise RuntimeError("Image decode failed")
                    
                    frame = cv2.flip(frame, -1)
                    
                    # Store for get_frame()
                    self.last_frame = frame
                    self.last_sensor = sensor
                    
                    hframe, wframe = frame.shape
                    img = frame[hframe//2:, :].flatten() / 255
                    
                    return img, sensor
        except socket.timeout:
            # Return last frame if available
            if self.last_frame is not None:
                hframe, wframe = self.last_frame.shape
                img = self.last_frame[hframe//2:, :].flatten() / 255
                return img, self.last_sensor
        except Exception as e:
            print(f"Error receiving frame: {e}")
            if self.last_frame is not None:
                hframe, wframe = self.last_frame.shape
                img = self.last_frame[hframe//2:, :].flatten() / 255
                return img, self.last_sensor
        
        # Fallback
        if self.last_frame is not None:
            hframe, wframe = self.last_frame.shape
            img = self.last_frame[hframe//2:, :].flatten() / 255
            return img, self.last_sensor
        
        raise RuntimeError("No frame received from Pi")
    
    def get_frame(self):
        return self.last_frame