import RPi.GPIO as GPIO
import pigpio

# motor pins
GPIO.setwarnings(False)
IN1, IN2, IN3, IN4 = 26, 19, 13, 6
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
RPWM1 = 5
RPWM2 = 21

GPIO.setup(RPWM1, GPIO.OUT)
p1 = GPIO.PWM(RPWM1, 50)

GPIO.setup(RPWM2, GPIO.OUT)
p2 = GPIO.PWM(RPWM2, 50)

# steering servo
pi = pigpio.pi()
servo_pin = 18

p1.start(100)
p2.start(100)

def set_servo(angle):
    pulse = 500 + (angle / 180) * 2000
    pi.set_servo_pulsewidth(servo_pin, pulse)

def backward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def forward():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

def stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)


def left():  set_servo(60)
def right(): set_servo(120)
def center(): set_servo(97)

