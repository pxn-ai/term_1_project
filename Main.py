"""
Smart Classroom Occupancy Counter
Detects movement via ultrasonic sensors, records video, and counts people entering/exiting.
"""

import os
import threading
from time import time, sleep
import cv2
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None
    print("Warning: picamera2 not found. PiCamera recording will not work.")

from gpiozero import LED, DistanceSensor
from Human_Identifier import HumanInOutCounter
from lcd_display import print_lcd_message, print_lcd_time
from eyes import show_happy, show_suspicious_left, show_suspicious_right, clear_face, show_buffering

detection_range = 200
USB_Camera_preferred = False
inside_classroom = 0
video_stack = []
frame_width, frame_height = 1280, 720

def record_picamera(wait_time=10):
    """Record video using PiCamera until no movement is detected for wait_time seconds."""
    if Picamera2 is None:
        print("Error: Picamera2 is not available.")
        return None

    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(main={"size": (frame_width, frame_height)})
    picam2.configure(video_config)

    start_time = time()
    print("Movement detected! Recording video...")
    video_filename = f"video_{int(time())}.h264"
    picam2.start_recording(video_filename)

    print_lcd_message("Movement detected!", "Recording video...")
    clear_face()
    looking_left = False
    while True:
        elapsed_time = time() - start_time
        time_remaining = wait_time - elapsed_time

        if is_human_present():
            if time_remaining <= wait_time:
                time_remaining = wait_time
        if time_remaining <= 0:
            clear_face()
            print_lcd_message("Recording stopped", " ")
            break

        if elapsed_time % 2 < 1:
            print_lcd_message(f"Recording... {int(elapsed_time)}s", " ")
            if looking_left:
                show_suspicious_left()
            else:
                show_suspicious_right()

        sleep(0.1)
    picam2.stop_recording()
    print(f"Video saved as {video_filename}")
    return video_filename
        
def record_usb_camera(wait_time=10):
    """Record video using USB camera until no movement is detected for wait_time seconds."""
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_filename = f"video_{int(time())}.avi"
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame_width, frame_height))

    start_time = time()
    print("Movement detected! Recording video...")
    print_lcd_message("Movement detected!", "Recording video...")
    clear_face()
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            out.write(frame)

        elapsed_time = time() - start_time
        time_remaining = wait_time - elapsed_time
        
        if is_human_present():
            if time_remaining <= wait_time:
                time_remaining = wait_time
        if time_remaining <= 0:
            print_lcd_message("Recording stopped", " ") 
            clear_face()
            break

        if elapsed_time % 2 < 1:
            print_lcd_message(f"Recording... {int(elapsed_time)}s", " ")
            if (elapsed_time // 2) % 2 == 0:
                show_suspicious_left()
            else:
                show_suspicious_right()
        sleep(1)
    out.release()
    cap.release()
    print(f"Video saved as {video_filename}")
    return video_filename
    
def is_human_present(left_sensor=None, right_sensor=None):
    """Check if a human is detected within range using ultrasonic sensors."""
    global ultrasonic_left, ultrasonic_right, detection_range
    
    sensor_l = left_sensor if left_sensor else ultrasonic_left
    sensor_r = right_sensor if right_sensor else ultrasonic_right
    
    distance_left = sensor_l.distance * 100
    distance_right = sensor_r.distance * 100
    return distance_left < detection_range or distance_right < detection_range

def analyze_video(video_filename, human_counter: HumanInOutCounter, args):
    """Analyze recorded video and return net count of people who entered."""
    print(f"Analyzing video {video_filename}...")
    print_lcd_message("Let me think ...", "Counting people.")
    clear_face()
    show_buffering(duration=120)

    net_count_in = human_counter.get_net_entered_count(
        video_path=video_filename,
        count_line_pos=args.line * frame_width
    )
    
    if args.json and net_count_in:
        human_counter.save_results(net_count_in, args.json)

    print_lcd_message("Analysis", f"complete =D")
    clear_face()
    show_happy()
    if net_count_in >= 0:
        print_lcd_message("Net People ", f"entered : {net_count_in}")
    else:
        print_lcd_message("Net People ", f"entered : {-net_count_in}")
    return net_count_in

def process_video_stack(human_counter, args):
    """Process queued videos and update classroom occupancy count."""
    global inside_classroom
    
    while len(video_stack) > 0:
        video_file = video_stack.pop(0)
        net_count = analyze_video(video_file, human_counter, args)
        print(f"Net people entered: {net_count}")
        inside_classroom += net_count
        if inside_classroom < 0:
            inside_classroom = 0

        if os.path.exists(video_file):
            os.remove(video_file)
        print(f"Current occupancy: {inside_classroom} people")
    
if __name__ == "__main__":
    import sys
    import argparse
    
    processing_thread = None

    parser = argparse.ArgumentParser(description='Video Human In/Out Counter')
    parser.add_argument('video', type=str, nargs='?', help='Path to video file (optional for live mode)')
    parser.add_argument('--output', type=str, default=None, help='Path to save annotated video')
    parser.add_argument('--preview', action='store_true', help='Show video preview while processing')
    parser.add_argument('--model', type=str, default='n', help='Model size: n (nano), s (small)')
    parser.add_argument('--skip', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--line', type=float, default=0.5, help='Counting line position (0.0-1.0)')
    parser.add_argument('--json', type=str, default=None, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    human_counter = HumanInOutCounter(model_size=args.model)
    print_lcd_message("Welcome to", "Smart Classroom")
    clear_face()
    show_happy()

    if args.video:
        analyze_video(args.video, human_counter, args)
        sys.exit(0)

    power = LED(17)
    ultrasonic_left = DistanceSensor(echo=27, trigger=22, max_distance=4)
    ultrasonic_right = DistanceSensor(echo=23, trigger=24, max_distance=4)
    inside_classroom = 2
    
    try:
        power.on()
        while True:
            if is_human_present():
                if not USB_Camera_preferred:
                    video_file = record_picamera(wait_time=5)
                else:
                    video_file = record_usb_camera(wait_time=5)

                video_stack.append(video_file)

                if processing_thread is None or not processing_thread.is_alive():
                    processing_thread = threading.Thread(target=process_video_stack, args=(human_counter, args))
                    processing_thread.start()

            sleep(0.05)
            if processing_thread is not None and not processing_thread.is_alive():
                if inside_classroom > 0:
                    power.on()
                else:
                    power.off()

                clear_face()
                show_happy()
                print_lcd_time(inside_classroom)

    except KeyboardInterrupt:
        print("Exiting program...")
        print_lcd_message("Shutting down", "Goodbye!")
        clear_face()
        show_happy()
        power.off()