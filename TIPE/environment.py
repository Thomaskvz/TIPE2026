import struct
import cv2


class Environment:
    def __init__(self, conn, connection):
        self.conn = conn
        self.connection = connection
        self.action_map={
            0:b'F',
            1:b'R',
            2:b'L'
        }


    def reset(self):
        self.conn.sendall(b"S")
        frame, sensor = self.process_image()
        return frame, sensor



    def step(self, action, count):
        self.conn.sendall(self.action_map[action])
        frame, sensor = self.process_image()
        done=False
        reward=1   
                
        if sensor!=b'00':
            count+=1
        if count>=2:
            done=True
            reward=-10

        return frame, sensor, reward, done, count
    

    def process_image(self):
        header = self.connection.read(struct.calcsize('<LL'))
        if not header:
            return
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

        sensor = b''
        while len(sensor) < capt_len:
            more = self.connection.read(capt_len - len(sensor))
            if not more:
                break
            sensor += more

        hframe, wframe = frame.shape

        img = frame/255
        img = img[hframe//2:,:].flatten()

        return img, sensor
