import socket
import RPi.GPIO as GPIO
import pigpio

# motor pins
IN1, IN2 = 22, 23
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

# steering servo
pi = pigpio.pi()
servo_pin = 23

def set_servo(angle):
    pulse = 500 + (angle / 180) * 2000
    pi.set_servo_pulsewidth(servo_pin, pulse)

def forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)

def backward():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)

def stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)

def left():  set_servo(60)
def right(): set_servo(120)
def center(): set_servo(90)

# listen for control commands
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("192.168.1.136", 9998))
print("Waiting for controller...")
conn, addr = s.accept()
print("Connected:", addr)

try:
    while True:
        cmd = conn.recv(1)
        if not cmd:
            break
        c = cmd.decode()
        if c == 'F': forward()
        elif c == 'B': backward()
        elif c == 'L': left()
        elif c == 'R': right()
        elif c == 'C': center()
        elif c == 'S': stop(); center()
finally:
    stop()
    center()
    pi.stop()
    GPIO.cleanup()
    s.close()

