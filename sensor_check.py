'''
Check the both Ultrasonic sensors and LEDs.
'''

from gpiozero import DistanceSensor, LED
from time import sleep

def LED_check(led):
    led.on()
    sleep(0.05)
    led.off()
    sleep(0.05)

if __name__ == "__main__":
    # Initialize Ultrasonic Sensors (max_distance=4 allows detection up to 400cm)
    ultrasonic_left = DistanceSensor(echo=27, trigger=22, max_distance=4)
    ultrasonic_right = DistanceSensor(echo=23, trigger=24, max_distance=4)

    # Initialize LEDs
    led = LED(17)
    try:
        while True:
            # Check LEDs
            LED_check(led)
            print("LED checked.")

            # Check Ultrasonic Sensors
            left_distance = ultrasonic_left.distance * 100  # Convert to cm
            right_distance = ultrasonic_right.distance * 100  # Convert to cm
            human_present = (left_distance < 200 or right_distance < 200)  # Example threshold for human presence
            print(
                f"Left Distance: {left_distance:.2f} cm  |  "
                f"Right Distance: {right_distance:.2f} cm  |  "
                f"Human Present: {human_present}"
            )
            sleep(0.1)  # Wait before next check

    except KeyboardInterrupt:
        print("Exiting sensor check.")