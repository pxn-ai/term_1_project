'''
When detected a movement through the Ultrasonic sensor, a video clip recorded and saved.
Then starts analyzing it and gets count of people went in and out of the classroom.
'''

import os
import threading
from time import time, sleep
import cv2
from picamera2 import Picamera2
from gpiozero import LED, UltrasonicSensor
from Human_Identifier import HumanInOutCounter  # Imports the Custom Model we built

detection_range = 20  # in cm
USB_Camera_preferred = True  # Set to False to use PiCamera instead of USB Camera
inside_classroom = 0  # Initial count of classroom occupancy
video_stack = []  # Stack of videos to be analyzed

def record_picamera( wait_time = 10 ):
    ''' Records a video clip until human movement is detected by Ultrasonic sensor.
        wait_time : duration of the wait in seconds.
    '''
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(main={"size": (1280, 720)})
    picam2.configure(video_config)

    start_time = time()
    while True:
        print("Movement detected! Recording video...")
        video_filename = f"video_{int(time())}.h264"
        picam2.start_recording(video_filename)

        while True:
            time_remaining = wait_time - (time() - start_time)

            if is_human_present() :  # If an object is detected within 100 cm
                if time_remaining <= wait_time:
                    time_remaining = wait_time  # Reset the timer if movement is detected
            if time_remaining <= 0:
                break
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
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (1280, 720))

    start_time = time()
    while True:
        print("Movement detected! Recording video...")
        while True:
            ret, frame = cap.read()
            if ret:
                out.write(frame)

            time_remaining = wait_time - (time() - start_time)
            
            if is_human_present() :  # If an object is detected within 100 cm
                if time_remaining <= wait_time:
                    time_remaining = wait_time  # Reset the timer if movement is detected
            if time_remaining <= 0:
                break
            sleep(0.1) # Small delay to prevent busy-waiting
        out.release()
        cap.release()
        print(f"Video saved as {video_filename}")
        return video_filename
    
def is_human_present() :
    ''' Checks if a human is present using the Ultrasonic sensors. '''
    global ultrasonic_left, ultrasonic_right, detection_range
    distance_left = ultrasonic_left.distance * 100  # Convert to cm
    distance_right = ultrasonic_right.distance * 100  # Convert to cm
    return distance_left < detection_range or distance_right < detection_range

def analyze_video( video_filename , human_counter : HumanInOutCounter ):
    ''' Analyzes the recorded video and returns count of people went in and out of the classroom.
        video_filename : path to the recorded video file.
    '''
    
    print(f"Analyzing video {video_filename}...")
    # Analyze video
    net_count_in = human_counter.get_net_entered_count(
        video_path=video_filename,
        output_path=args.output,
        show_preview=args.preview,
        skip_frames=args.skip,
        count_line_pos=args.line
    )
    
    # Save results if requested
    if args.json and net_count_in :
        human_counter.save_results(net_count_in, args.json)

    return net_count_in

def process_video_stack( human_counter ):
    ''' Processes videos in the stack one by one. '''
    global inside_classroom
    
    while len(video_stack) > 0:
        video_file = video_stack.pop(0)
        net_count = analyze_video(video_file, human_counter)
        print(f"Net people entered: {net_count}")
        inside_classroom += net_count
        if inside_classroom < 0:
            inside_classroom = 0  # Prevent negative count

        # delete the analyzed video file to save space
        os.remove(video_file)
        print(f"Current occupancy: {inside_classroom} people")
    
if __name__ == "__main__":
    import sys
    import argparse
    
    processing_thread = None

    parser = argparse.ArgumentParser(description='Video Human In/Out Counter')
    parser.add_argument('video', type=str, help='Path to video file')
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

    power = LED(17)  # LED for indicating classroom power status
    ultrasonic_left = UltrasonicSensor(echo=27, trigger=22)  # Ultrasonic sensor for movement detection
    ultrasonic_right = UltrasonicSensor(echo=5, trigger=6)

    while True:
        
        if is_human_present() :  # If an object is detected within 100 cm
            if not USB_Camera_preferred:
                video_file = record_picamera( wait_time=10)
            else:
                video_file = record_usb_camera( wait_time=10)

            # Analyze recorded video

            video_stack.append(video_file)

            if processing_thread is None or not processing_thread.is_alive():
                processing_thread = threading.Thread(target=process_video_stack, args=(human_counter,))
                processing_thread.start()

        sleep(0.5)  # Small delay to prevent busy-waiting
        if processing_thread is not None and not processing_thread.is_alive():
            if inside_classroom > 0:
                power.on()  # Turn on power if there are people inside
            else:
                power.off()  # Turn off power if no one is inside
