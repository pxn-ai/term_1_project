"""LED Matrix faces with scrolling text for MAX7219 display."""

import time
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.led_matrix.device import max7219
from luma.core.legacy import show_message
from luma.core.legacy.font import proportional, CP437_FONT, TINY_FONT

serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=1, block_orientation=180)


def show_happy():
    """Display happy face ( ^ _ ^ )."""
    with canvas(device) as draw:
        draw.point((1, 3), fill="white"); draw.point((2, 2), fill="white"); draw.point((3, 3), fill="white")
        draw.point((5, 3), fill="white"); draw.point((6, 2), fill="white"); draw.point((7, 3), fill="white")
        draw.line((3, 6, 4, 6), fill="white")


def show_suspicious():
    """Display suspicious face ( < _ < )."""
    with canvas(device) as draw:
        draw.point((3, 2), fill="white"); draw.point((2, 3), fill="white"); draw.point((3, 4), fill="white")
        draw.point((7, 2), fill="white"); draw.point((6, 3), fill="white"); draw.point((7, 4), fill="white")
        draw.line((3, 6, 4, 6), fill="white")


def scroll_text(msg):
    """Scroll text across the LED matrix."""
    show_message(device, msg, fill="white", font=proportional(CP437_FONT), scroll_delay=0.08)


print("Running... Press Ctrl+C to stop.")

try:
    while True:
        scroll_text("HELLO!")
        time.sleep(0.5)

        show_happy()
        time.sleep(2)

        scroll_text("I AM WATCHING")
        
        show_suspicious()
        time.sleep(2)

except KeyboardInterrupt:
    device.cleanup()
    print("Goodbye!")