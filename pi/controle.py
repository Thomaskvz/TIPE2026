import RPi.GPIO as GPIO
import pigpio

# motor pins
GPIO.setwarnings(False)
IN1, IN2 = 19, 26
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

def left():  set_servo(50)
def right(): set_servo(130)
def center(): set_servo(90)

