"""Test script to verify both ultrasonic sensors and LED functionality."""

from gpiozero import DistanceSensor, LED
from time import sleep


def LED_check(led):
    """Flash LED briefly to verify it works."""
    led.on()
    sleep(0.05)
    led.off()
    sleep(0.05)


if __name__ == "__main__":
    ultrasonic_left = DistanceSensor(echo=27, trigger=22, max_distance=4)
    ultrasonic_right = DistanceSensor(echo=23, trigger=24, max_distance=4)
    led = LED(17)
    
    try:
        while True:
            LED_check(led)
            print("LED checked.")

            left_distance = ultrasonic_left.distance * 100
            right_distance = ultrasonic_right.distance * 100
            human_present = (left_distance < 200 or right_distance < 200)
            print(
                f"Left Distance: {left_distance:.2f} cm  |  "
                f"Right Distance: {right_distance:.2f} cm  |  "
                f"Human Present: {human_present}"
            )
            sleep(0.1)

    except KeyboardInterrupt:
        print("Exiting sensor check.")