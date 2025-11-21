'''
When detected a movement through the Ultrasonic sensor, a video clip recorded and saved.
Then starts analyzing it and gets count of people went in and out of the classroom.
'''

from time import time
import cv2
from picamera2 import Picamera2
from gpiozero import LED, UltrasonicSensor

