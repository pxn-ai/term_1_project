"""Test script for single ultrasonic sensor distance measurement."""

from gpiozero import DistanceSensor
from time import sleep

ultrasonic = DistanceSensor(echo=23, trigger=24)

print("Ultrasonic Sensor Test")
print("Press Ctrl+C to exit")

try:
    while True:
        distance_cm = ultrasonic.distance * 100
        print(f"Distance: {distance_cm:.2f} cm")
        sleep(0.01)
except KeyboardInterrupt:
    print("\nExiting...")
