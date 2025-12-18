"""LED Matrix facial expressions for 8x8 MAX7219 display."""

import time
import random
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.led_matrix.device import max7219

serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=1, block_orientation=0)


def show_happy():
    """Display happy expression ( ^ _ ^ )."""
    with canvas(device) as draw:
        draw.point((0, 2), fill="white")
        draw.point((1, 1), fill="white")
        draw.point((2, 2), fill="white")
        
        draw.point((5, 2), fill="white")
        draw.point((6, 1), fill="white")
        draw.point((7, 2), fill="white")
        
        draw.line((3, 7, 4, 7), fill="white")
        draw.point((2, 6), fill="white")
        draw.point((5, 6), fill="white")


def show_suspicious_left():
    """Display suspicious left expression ( < _ < )."""
    with canvas(device) as draw:
        draw.point((2, 1), fill="white")
        draw.point((1, 2), fill="white")
        draw.point((2, 3), fill="white")

        draw.point((6, 1), fill="white")
        draw.point((5, 2), fill="white")
        draw.point((6, 3), fill="white")

        draw.line((3, 7, 4, 7), fill="white")
        draw.point((2, 6), fill="white")
        draw.point((5, 6), fill="white")


def show_suspicious_right():
    """Display suspicious right expression ( > _ > )."""
    with canvas(device) as draw:
        draw.point((1, 1), fill="white")
        draw.point((2, 2), fill="white")
        draw.point((1, 3), fill="white")

        draw.point((5, 1), fill="white")
        draw.point((6, 2), fill="white")
        draw.point((5, 3), fill="white")

        draw.line((3, 7, 4, 7), fill="white")
        draw.point((2, 6), fill="white")
        draw.point((5, 6), fill="white")


def clear_face():
    """Clear the LED matrix display."""
    with canvas(device) as draw:
        pass


def show_buffering(duration=10.0, speed=0.1):
    """Display rotating loading animation for specified duration."""
    frames = [
        [(3, 0), (4, 0)],
        [(5, 1), (6, 2)],
        [(7, 3), (7, 4)],
        [(6, 5), (5, 6)],
        [(4, 7), (3, 7)],
        [(2, 6), (1, 5)],
        [(0, 4), (0, 3)],
        [(1, 2), (2, 1)],
    ]
    
    start_time = time.time()
    frame_index = 0
    
    clear_face()
    while time.time() - start_time < duration:
        with canvas(device) as draw:
            draw.point((3, 3), fill="white")
            draw.point((4, 3), fill="white")
            draw.point((3, 4), fill="white")
            draw.point((4, 4), fill="white")
            
            for i in range(3):
                idx = (frame_index - i) % len(frames)
                for point in frames[idx]:
                    draw.point(point, fill="white")
        
        frame_index = (frame_index + 1) % len(frames)
        time.sleep(speed)


if __name__ == "__main__":
    print("Displaying cute faces... Press Ctrl+C to stop.")

    try:
        while True:
            show_happy()
            time.sleep(2)
            
            show_suspicious_left()
            time.sleep(1)
            
            show_suspicious_right()
            time.sleep(1)
            
            clear_face()
            time.sleep(0.2)

            show_buffering(duration=4.0, speed=0.1)

    except KeyboardInterrupt:
        device.cleanup()
        print("Goodbye!")