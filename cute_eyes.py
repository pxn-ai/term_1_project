import time
import random
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.led_matrix.device import max7219

# --- SETUP ---
# We are using SPI port 0, device 0. 
# 'cascaded=1' means you have 1 LED block. If you chain them later, change this number.
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=1, block_orientation=0) # Orientation might need adjustment (0, 90, -90) based on how you hold it

# --- ANIMATION FUNCTIONS ---

def draw_eyes(draw, look_x=0, look_y=0, blink=False, happy=False):
    """
    Draws two eyes on the 8x8 grid.
    look_x: -1 (left), 0 (center), 1 (right)
    look_y: -1 (up), 0 (center), 1 (down)
    blink: True to close eyes
    happy: True for ^ ^ eyes
    """
    
    # Eye positions (Left Eye, Right Eye)
    # Format: (x, y) top-left corner of the eye box
    left_eye_origin = (1, 1)
    right_eye_origin = (5, 1)
    
    if blink:
        # Draw closed eyes (flat lines)
        draw.line((1, 3, 3, 3), fill="white")
        draw.line((5, 3, 7, 3), fill="white")
        return

    if happy:
        # Draw Happy Eyes (inverted V shape like ^ ^)
        # Left Eye
        draw.point((1, 3), fill="white")
        draw.point((2, 2), fill="white")
        draw.point((3, 3), fill="white")
        # Right Eye
        draw.point((5, 3), fill="white")
        draw.point((6, 2), fill="white")
        draw.point((7, 3), fill="white")
        return

    # --- Normal Open Eyes ---
    # Draw Eye Whites (3x3 boxes)
    draw.rectangle((1, 1, 3, 4), outline="white", fill="black")
    draw.rectangle((5, 1, 7, 4), outline="white", fill="black")
    
    # Draw Pupils (1 pixel dot)
    # Base pupil position is center of the 3x3 box
    pupil_l_x = 2 + look_x
    pupil_l_y = 2 + look_y
    
    pupil_r_x = 6 + look_x
    pupil_r_y = 2 + look_y
    
    draw.point((pupil_l_x, pupil_l_y), fill="white")
    draw.point((pupil_r_x, pupil_r_y), fill="white")


def animate_blink():
    # Quick Blink
    with canvas(device) as draw:
        draw_eyes(draw, blink=True)
    time.sleep(0.1)
    with canvas(device) as draw:
        draw_eyes(draw, look_x=0, look_y=0)
    time.sleep(random.uniform(0.5, 2.0))

def peek_left():
    # Look center -> move pupil left -> hold -> back center
    with canvas(device) as draw:
        draw_eyes(draw, look_x=-1) # Look Left
    time.sleep(0.8)
    with canvas(device) as draw:
        draw_eyes(draw, look_x=0) # Center
    time.sleep(0.5)

def peek_right():
    # Look center -> move pupil right -> hold -> back center
    with canvas(device) as draw:
        draw_eyes(draw, look_x=1) # Look Right
    time.sleep(0.8)
    with canvas(device) as draw:
        draw_eyes(draw, look_x=0) # Center
    time.sleep(0.5)

def act_happy():
    # Show happy eyes ^ ^
    with canvas(device) as draw:
        draw_eyes(draw, happy=True)
    time.sleep(1.5)


# --- MAIN LOOP ---
print("Press Ctrl+C to stop animations.")
try:
    while True:
        # Pick a random action
        action = random.choice(['blink', 'left', 'right', 'happy', 'blink'])
        
        if action == 'blink':
            animate_blink()
        elif action == 'left':
            peek_left()
        elif action == 'right':
            peek_right()
        elif action == 'happy':
            act_happy()
            
        time.sleep(0.2)

except KeyboardInterrupt:
    # Clear display on exit
    device.cleanup()
    print("Goodbye!")