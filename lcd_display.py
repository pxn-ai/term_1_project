#! /home/pasanrk/Documents/Python/python_venv/bin/python

'''LCD Display Module with Raspberry Pi'''

from datetime import datetime
import time
from RPLCD.i2c import CharLCD
import time

# Initialize the LCD display
lcd = CharLCD('PCF8574', 0x27)

def print_lcd_message(line1, line2):
    '''Prints two lines of message on the LCD display.'''
    lcd.clear()
    lcd.write_string(line1)
    lcd.cursor_pos = (1, 0)
    lcd.write_string(line2)

def print_lcd_time():
    '''Prints the current date and time on the LCD display.'''
    now = datetime.now()
    date_str = now.date().strftime("%Y-%m-%d")
    time_str = now.time().strftime("%H:%M:%S")
    
    lcd.clear()
    lcd.write_string(date_str)
    lcd.cursor_pos = (1, 0)
    lcd.write_string(time_str)

if __name__ == "__main__":
    try :
        # Clear the display
        lcd.clear()

        # Write a message to the LCD
        lcd.write_string('Hello, World!')

        # Wait for 2 seconds
        time.sleep(2)

        # Move to the second line and write another message
        lcd.cursor_pos = (1, 0)
        lcd.write_string('Raspberry Pi LCD')

        # Wait for 150 seconds before clearing
        time.sleep(10)

        # starting time
        start_time = time.time()
        while True:
            # Clear the display
            lcd.clear()
            # Get the current date and time
            now = datetime.now()
            # Format the date and time
            date_str = now.date().strftime("%Y-%m-%d")
            time_str = now.time().strftime("%H:%M:%S")
            # Display the date and time
            lcd.write_string(date_str)
            lcd.cursor_pos = (1, 0)
            lcd.write_string(time_str)
            # Wait for 1 second before updating
            time.sleep(1)

            # Check if 150 seconds have passed
            if time.time() - start_time >= 150:
                # Clear the display
                lcd.clear()
                break

        lcd.clear()

    except KeyboardInterrupt:
        lcd.clear()

    finally:
        # Clean up and close the LCD
        lcd.close()
        print("LCD display closed.")