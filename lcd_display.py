#!/home/pasanrk/Documents/Python/python_venv/bin/python
"""LCD Display module for 16x2 I2C character LCD."""

from datetime import datetime
import time
from RPLCD.i2c import CharLCD

lcd = CharLCD('PCF8574', 0x27)


def print_lcd_message(line1, line2):
    """Display two lines of text on the LCD."""
    lcd.clear()
    lcd.write_string(line1)
    lcd.cursor_pos = (1, 0)
    lcd.write_string(line2)


def print_lcd_time(count_in: int):
    """Display current date, time, and classroom occupancy count."""
    now = datetime.now()
    date_str = now.date().strftime("%m-%d")
    time_str = now.time().strftime("%H:%M")
    
    lcd.clear()
    lcd.write_string(date_str + " " + time_str)
    lcd.cursor_pos = (1, 0)
    lcd.write_string(f"In Class: {count_in}")


if __name__ == "__main__":
    try:
        lcd.clear()
        lcd.write_string('Hello, World!')
        time.sleep(2)

        lcd.cursor_pos = (1, 0)
        lcd.write_string('Raspberry Pi LCD')
        time.sleep(10)

        start_time = time.time()
        while True:
            lcd.clear()
            now = datetime.now()
            date_str = now.date().strftime("%Y-%m-%d")
            time_str = now.time().strftime("%H:%M:%S")
            lcd.write_string(date_str)
            lcd.cursor_pos = (1, 0)
            lcd.write_string(time_str)
            time.sleep(1)

            if time.time() - start_time >= 150:
                lcd.clear()
                break

        lcd.clear()

    except KeyboardInterrupt:
        lcd.clear()

    finally:
        lcd.close()
        print("LCD display closed.")