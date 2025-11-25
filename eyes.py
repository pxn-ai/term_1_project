import time
import random
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.led_matrix.device import max7219

# --- SETUP ---
# Remember to keep block_orientation at 0, 90, or -90 based on what worked for you last time!
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=1, block_orientation=0)

# --- DRAWING FUNCTIONS ---

def show_happy():
    # Expression: ( ^ _ ^ )
    with canvas(device) as draw:
        # Left Eye (^)
        draw.point((0, 2), fill="white")
        draw.point((1, 1), fill="white")
        draw.point((2, 2), fill="white")
        
        # Right Eye (^)
        draw.point((5, 2), fill="white")
        draw.point((6, 1), fill="white")
        draw.point((7, 2), fill="white")
        
        # Mouth (_) - A small line at the bottom
        draw.line((3, 7, 4, 7), fill="white")
        draw.point((2, 6), fill="white")
        draw.point((5, 6), fill="white")

def show_suspicious_left():
    # Expression: ( < _ < )
    with canvas(device) as draw:
        # Left Eye (<)
        # Points: Top-right, Middle-left, Bottom-right
        draw.point((2, 1), fill="white")
        draw.point((1, 2), fill="white")
        draw.point((2, 3), fill="white")

        # Right Eye (<)
        draw.point((6, 1), fill="white")
        draw.point((5, 2), fill="white")
        draw.point((6, 3), fill="white")

        # Mouth (_)
        draw.line((3, 7, 4, 7), fill="white")
        draw.point((2, 6), fill="white")
        draw.point((5, 6), fill="white")

def show_suspicious_right():
    # Expression: ( > _ > )
    with canvas(device) as draw:
        # Left Eye (>)
        # Points: Top-left, Middle-right, Bottom-left
        draw.point((1, 1), fill="white")
        draw.point((2, 2), fill="white")
        draw.point((1, 3), fill="white")

        # Right Eye (>)
        draw.point((5, 1), fill="white")
        draw.point((6, 2), fill="white")
        draw.point((5, 3), fill="white")

        # Mouth (_)
        draw.line((3, 7, 4, 7), fill="white")
        draw.point((2, 6), fill="white")
        draw.point((5, 6), fill="white")

def clear_face():
    with canvas(device) as draw:
        pass # Draws nothing, clears screen

def show_buffering(duration=10.0, speed=0.1):
    """
    Display a rotating buffering/loading animation.
    duration: Total time to show the animation in seconds.
    speed: Time between each frame in seconds.
    """
    # Define the 8 frames of a rotating spinner (clockwise)
    # Each frame is a list of (x, y) points to light up
    frames = [
        [(3, 0), (4, 0)],                    # Top
        [(5, 1), (6, 2)],                    # Top-right
        [(7, 3), (7, 4)],                    # Right
        [(6, 5), (5, 6)],                    # Bottom-right
        [(4, 7), (3, 7)],                    # Bottom
        [(2, 6), (1, 5)],                    # Bottom-left
        [(0, 4), (0, 3)],                    # Left
        [(1, 2), (2, 1)],                    # Top-left
    ]
    
    start_time = time.time()
    frame_index = 0
    
    while time.time() - start_time < duration:
        with canvas(device) as draw:
            # Draw center dot
            draw.point((3, 3), fill="white")
            draw.point((4, 3), fill="white")
            draw.point((3, 4), fill="white")
            draw.point((4, 4), fill="white")
            
            # Draw the current spinner segment and the two before it (trail effect)
            for i in range(3):
                idx = (frame_index - i) % len(frames)
                for point in frames[idx]:
                    draw.point(point, fill="white")
        
        frame_index = (frame_index + 1) % len(frames)
        time.sleep(speed)

if __name__ == "__main__":
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

            # 4. Buffering Animation
            show_buffering(duration=4.0, speed=0.1)

    except KeyboardInterrupt:
        device.cleanup()
        print("Goodbye!")