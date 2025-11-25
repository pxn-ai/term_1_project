'''
When detected a movement through the Ultrasonic sensor, a video clip recorded and saved.
Then starts analyzing it and gets count of people went in and out of the classroom.
'''

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
from Human_Identifier import HumanInOutCounter  # Imports the Custom Model we built

from lcd_display import print_lcd_message, print_lcd_message
from eyes import show_happy, show_suspicious_left, show_suspicious_right, clear_face, show_buffering

detection_range = 200  # in cm
USB_Camera_preferred = True  # Set to False to use PiCamera instead of USB Camera
inside_classroom = 0  # Initial count of classroom occupancy
video_stack = []  # Stack of videos to be analyzed
frame_width, frame_height = 1280 , 720 # Width of the video frame

def record_picamera( wait_time = 10 ):
    ''' Records a video clip until human movement is detected by Ultrasonic sensor.
        wait_time : duration of the wait in seconds.
    '''
    if Picamera2 is None:
        print("Error: Picamera2 is not available.")
        return None

    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(main={"size": (frame_width, frame_height)})
    picam2.configure(video_config)

    start_time = time()
    # Removed redundant outer while True loop
    print("Movement detected! Recording video...")
    video_filename = f"video_{int(time())}.h264"
    picam2.start_recording(video_filename)

    print_lcd_message("Movement detected!", "Recording video...")
    clear_face()
    looking_left = False
    while True:
        elapsed_time = time() - start_time
        time_remaining = wait_time - elapsed_time

        if is_human_present() :  # If an object is detected within 100 cm
            if time_remaining <= wait_time:
                time_remaining = wait_time  # Reset the timer if movement is detected
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

        sleep(0.1) # Small delay to prevent busy-waiting
    picam2.stop_recording()
    print(f"Video saved as {video_filename}")
    return video_filename
        
def record_usb_camera( wait_time = 10 ):
    ''' Records a video clip until human movement is detected by Ultrasonic sensor.
        wait_time : duration of the wait in seconds.
    '''
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
            # Rotate frame 180 degrees before saving. before the camera is fixed upside down
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            out.write(frame)

        elapsed_time = time() - start_time
        time_remaining = wait_time - elapsed_time
        
        if is_human_present() :  # If an object is detected within 
            if time_remaining <= wait_time:
                time_remaining = wait_time  # Reset the timer if movement is detected
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
        sleep(0.1) # Small delay to prevent busy-waiting
    out.release()
    cap.release()
    print(f"Video saved as {video_filename}")
    return video_filename
    
def is_human_present(left_sensor=None, right_sensor=None) :
    ''' Checks if a human is present using the Ultrasonic sensors. '''
    global ultrasonic_left, ultrasonic_right, detection_range
    
    sensor_l = left_sensor if left_sensor else ultrasonic_left
    sensor_r = right_sensor if right_sensor else ultrasonic_right
    
    distance_left = sensor_l.distance * 100  # Convert to cm
    distance_right = sensor_r.distance * 100  # Convert to cm
    return distance_left < detection_range or distance_right < detection_range

def analyze_video( video_filename , human_counter : HumanInOutCounter, args ):
    ''' Analyzes the recorded video and returns count of people went in and out of the classroom.
        video_filename : path to the recorded video file.
    '''
    
    print(f"Analyzing video {video_filename}...")
    print_lcd_message("Let me think ...", "Counting your movements")
    clear_face()
    show_buffering()
    # Analyze video
    net_count_in = human_counter.get_net_entered_count(
        video_path=video_filename,
        count_line_pos=args.line * frame_width
    )
    
    # Save results if requested
    if args.json and net_count_in :
        human_counter.save_results(net_count_in, args.json)

    print_lcd_message("Analysis complete", f"Net People entered: {net_count_in}" if net_count_in >= 0 else f"Net People left: {-net_count_in}")
    clear_face()
    show_happy() if net_count_in >= 0 else show_suspicious_left()
    return net_count_in

def process_video_stack( human_counter, args ):
    ''' Processes videos in the stack one by one. '''
    global inside_classroom
    
    while len(video_stack) > 0:
        video_file = video_stack.pop(0)
        net_count = analyze_video(video_file, human_counter, args)
        print(f"Net people entered: {net_count}")
        inside_classroom += net_count
        if inside_classroom < 0:
            inside_classroom = 0  # Prevent negative count

        # delete the analyzed video file to save space
        if os.path.exists(video_file):
            os.remove(video_file)
        print(f"Current occupancy: {inside_classroom} people")
    
if __name__ == "__main__":
    import sys
    import argparse
    
    processing_thread = None

    parser = argparse.ArgumentParser(description='Video Human In/Out Counter')
    # Made video argument optional (nargs='?') so script runs without it
    parser.add_argument('video', type=str, nargs='?', help='Path to video file (optional for live mode)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save annotated video (optional)')
    parser.add_argument('--preview', action='store_true',
                       help='Show video preview while processing')
    parser.add_argument('--model', type=str, default='n',
                       help='Model size: n (nano), s (small)')
    parser.add_argument('--skip', type=int, default=1,
                       help='Process every Nth frame (default: 2)')
    parser.add_argument('--line', type=float, default=0.5,
                       help='Counting line position 0.0-1.0 from left (default: 0.5 = middle)')
    parser.add_argument('--json', type=str, default=None,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Create counter
    human_counter = HumanInOutCounter(model_size=args.model)
    # Print initial message
    print_lcd_message("Welcome to", "Smart Classroom")
    clear_face()
    show_happy()
    # If a video file is provided via CLI, analyze it immediately and exit
    if args.video:
        analyze_video(args.video, human_counter, args)
        sys.exit(0)

    power = LED(17)  # LED for indicating classroom power status
    ultrasonic_left = DistanceSensor(echo=27, trigger=22, max_distance=4)  # Ultrasonic sensors for movement detection
    ultrasonic_right = DistanceSensor(echo=23, trigger=24, max_distance=4)

    while True:
        
        if is_human_present() :  # If an object is detected within 100 cm
            if not USB_Camera_preferred:
                video_file = record_picamera( wait_time=5)
            else:
                video_file = record_usb_camera( wait_time=5)

            # Analyze recorded video

            video_stack.append(video_file)

            if processing_thread is None or not processing_thread.is_alive():
                # Pass args to the thread
                processing_thread = threading.Thread(target=process_video_stack, args=(human_counter, args))
                processing_thread.start()

        sleep(0.05)  # Small delay to prevent busy-waiting
        if processing_thread is not None and not processing_thread.is_alive():
            if inside_classroom > 0:
                power.on()  # Turn on power if there are people inside
            else:
                power.off()  # Turn off power if no one is inside
