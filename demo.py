from eyes import show_buffering, show_suspicious_right, show_suspicious_left, clear_face, show_happy
from time import sleep
from lcd_display import print_lcd_message, print_lcd_time
from gpiozero import LED

power = LED(17)  # GPIO pin to control power


power.off()
print_lcd_message("Let me think ...", "Counting people.")
show_buffering(duration=2)
print_lcd_message("Net people ", f"entered : 1")
show_happy()
power.on()  # Turn on power initially

sleep(2)

print_lcd_message("Let me think ...", "Counting people.")
show_buffering(duration=2)
print_lcd_message("Net people ", f"exited : 1")
show_suspicious_right()
power.off()
sleep(2)