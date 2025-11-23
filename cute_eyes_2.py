import time
import random
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.led_matrix.device import max7219

# --- SETUP ---
# Remember to keep block_orientation at 0, 90, or -90 based on what worked for you last time!
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=1, block_orientation=180)

# --- DRAWING FUNCTIONS ---

def show_happy():
    # Expression: ( ^ _ ^ )
    with canvas(device) as draw:
        # Left Eye (^)
        draw.point((1, 3), fill="white")
        draw.point((2, 2), fill="white")
        draw.point((3, 3), fill="white")
        
        # Right Eye (^)
        draw.point((5, 3), fill="white")
        draw.point((6, 2), fill="white")
        draw.point((7, 3), fill="white")
        
        # Mouth (_) - A small line at the bottom
        draw.line((3, 6, 4, 6), fill="white")

def show_suspicious_left():
    # Expression: ( < _ < )
    with canvas(device) as draw:
        # Left Eye (<)
        # Points: Top-right, Middle-left, Bottom-right
        draw.point((3, 2), fill="white")
        draw.point((2, 3), fill="white")
        draw.point((3, 4), fill="white")

        # Right Eye (<)
        draw.point((7, 2), fill="white")
        draw.point((6, 3), fill="white")
        draw.point((7, 4), fill="white")

        # Mouth (_)
        draw.line((3, 6, 4, 6), fill="white")

def show_suspicious_right():
    # Expression: ( > _ > )
    with canvas(device) as draw:
        # Left Eye (>)
        # Points: Top-left, Middle-right, Bottom-left
        draw.point((1, 2), fill="white")
        draw.point((2, 3), fill="white")
        draw.point((1, 4), fill="white")

        # Right Eye (>)
        draw.point((5, 2), fill="white")
        draw.point((6, 3), fill="white")
        draw.point((5, 4), fill="white")

        # Mouth (_)
        draw.line((3, 6, 4, 6), fill="white")

def clear_face():
    with canvas(device) as draw:
        pass # Draws nothing, clears screen

# --- MAIN ANIMATION LOOP ---
print("Displaying cute faces... Press Ctrl+C to stop.")

try:
    while True:
        # 1. Happy ( ^ _ ^ )
        show_happy()
        time.sleep(2)
        
        # 2. Suspicious Left ( < _ < )
        show_suspicious_left()
        time.sleep(1)
        
        # 3. Suspicious Right ( > _ > )
        show_suspicious_right()
        time.sleep(1)
        
        # Blink effect (Clear screen briefly)
        clear_face()
        time.sleep(0.2)

except KeyboardInterrupt:
    device.cleanup()
    print("Goodbye!")