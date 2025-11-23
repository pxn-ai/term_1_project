import time
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.led_matrix.device import max7219

# --- NEW IMPORTS FOR TEXT ---
from luma.core.legacy import show_message
from luma.core.legacy.font import proportional, CP437_FONT, TINY_FONT

# --- SETUP ---
serial = spi(port=0, device=0, gpio=noop())
# Remember to check your block_orientation (0, 90, -90)
device = max7219(serial, cascaded=1, block_orientation=-90)

# --- FACE FUNCTIONS (From before) ---
def show_happy():
    with canvas(device) as draw:
        # ^ _ ^
        draw.point((1, 3), fill="white"); draw.point((2, 2), fill="white"); draw.point((3, 3), fill="white")
        draw.point((5, 3), fill="white"); draw.point((6, 2), fill="white"); draw.point((7, 3), fill="white")
        draw.line((3, 6, 4, 6), fill="white")

def show_suspicious():
    with canvas(device) as draw:
        # < _ <
        draw.point((3, 2), fill="white"); draw.point((2, 3), fill="white"); draw.point((3, 4), fill="white")
        draw.point((7, 2), fill="white"); draw.point((6, 3), fill="white"); draw.point((7, 4), fill="white")
        draw.line((3, 6, 4, 6), fill="white")

# --- TEXT FUNCTION ---
def scroll_text(msg):
    # scroll_delay: lower number = faster speed (0.05 is fast, 0.1 is normal)
    # font: CP437_FONT is the standard retro pixel font
    show_message(device, msg, fill="white", font=proportional(CP437_FONT), scroll_delay=0.08)

# --- MAIN LOOP ---
print("Running... Press Ctrl+C to stop.")

try:
    while True:
        # 1. Scroll Hello
        scroll_text("HELLO!")
        time.sleep(0.5)

        # 2. Show Happy Face
        show_happy()
        time.sleep(2)

        # 3. Scroll user status
        scroll_text("I AM WATCHING")
        
        # 4. Show Suspicious Face
        show_suspicious()
        time.sleep(2)

except KeyboardInterrupt:
    device.cleanup()
    print("Goodbye!")