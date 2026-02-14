import RPi.GPIO as GPIO
import time

led_pin = 17        # GPIO pin for LED
pir_pin = 2        # GPIO pin for PIR sensor
timeout = 10       # seconds
last_motion_time = 0

GPIO.setmode(GPIO.BCM)         # Use BCM numbering (GPIO numbers)
GPIO.setup(led_pin, GPIO.OUT)  # Set LED as output
GPIO.setup(pir_pin, GPIO.IN)   # Set PIR as input

print("PIR Sensor Active... (Press Ctrl+C to stop)")

try:
    while True:
        motion = GPIO.input(pir_pin)
        if motion == 1:
            GPIO.output(led_pin, GPIO.HIGH)
            last_motion_time = time.time()
            print("Motion detected!")
        else:
            # Check if no motion for 'timeout' seconds
            if time.time() - last_motion_time > timeout:
                GPIO.output(led_pin, GPIO.LOW)
                print("No motion for", timeout, "seconds - LED OFF")

        time.sleep(0.2)  # Small delay to reduce CPU usage

except KeyboardInterrupt:
    print("Exiting program...")

finally:
    GPIO.cleanup()