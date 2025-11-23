'''
Check the both Ultrasonic sensors and LEDs.
'''

from gpiozero import DistanceSensor, LED
from time import sleep
from Main import is_human_present

def LED_check(led):
    led.on()
    sleep(0.5)
    led.off()
    sleep(0.5)

def ultrasonic_check(sensor):
    distance = sensor.distance * 100  # Convert to centimeters
    return distance

if __name__ == "__main__":
    # Initialize Ultrasonic Sensors
    ultrasonic_left = DistanceSensor(echo=27, trigger=22)
    ultrasonic_right = DistanceSensor(echo=5, trigger=6)

    # Initialize LEDs
    led = LED(17)

    try:
        while True:
            # Check LEDs
            LED_check(led)
            print("LED checked.")

            # Check Ultrasonic Sensors
            left_distance = ultrasonic_check(ultrasonic_left)
            right_distance = ultrasonic_check(ultrasonic_right)
            human_present = is_human_present(ultrasonic_left, ultrasonic_right)
            print(
                f"Left Distance: {left_distance:.2f} cm  |  "
                f"Right Distance: {right_distance:.2f} cm  |  "
                f"Human Present: {human_present}"
            )
            sleep(1)  # Wait before next check

    except KeyboardInterrupt:
        print("Exiting sensor check.")