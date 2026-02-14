import cv2
import numpy as np
from datetime import datetime

def capture_and_detect_person():
    """
    Capture a photo from webcam, detect persons, draw rectangles around them, and display the result.
    """
    # Initialize the camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Allow camera to warm up
    print("Camera ready. Press SPACE to capture photo or 'q' to quit")
    
    # Initialize HOG (Histogram of Oriented Gradients) person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Display live feed
        cv2.imshow('Live Feed - Press SPACE to capture, Q to quit', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space bar pressed
            print("\nCapturing photo...")
            
            # Save the original photo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = f"captured_photo_{timestamp}.jpg"
            cv2.imwrite(original_filename, frame)
            print(f"Original photo saved as: {original_filename}")
            
            # Detect persons in the captured frame
            print("Detecting persons...")
            
            # Detect people using HOG descriptor
            # detectMultiScale parameters: (image, winStride, padding, scale)
            boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
            
            # Draw rectangles around detected persons
            detected_frame = frame.copy()
            person_count = 0
            
            for (x, y, w, h) in boxes:
                person_count += 1
                # Draw rectangle (green color, thickness 3)
                cv2.rectangle(detected_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # Add label
                cv2.putText(detected_frame, f'Person {person_count}', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            print(f"Detected {person_count} person(s)")
            
            # Save the annotated photo
            annotated_filename = f"detected_persons_{timestamp}.jpg"
            cv2.imwrite(annotated_filename, detected_frame)
            print(f"Annotated photo saved as: {annotated_filename}")
            
            # Display both images
            cv2.imshow('Original Photo', frame)
            cv2.imshow('Detected Persons', detected_frame)
            
            print("\nPress any key to continue capturing or 'q' to quit")
            
        elif key == ord('q'):  # 'q' pressed
            print("\nExiting...")
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def detect_person_in_image(image_path):
    """
    Detect persons in an existing image file.
    
    Args:
        image_path: Path to the image file
    """
    # Read the image
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    # Initialize HOG person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Detect persons
    print("Detecting persons...")
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
    
    # Draw rectangles around detected persons
    detected_frame = frame.copy()
    person_count = 0
    
    for (x, y, w, h) in boxes:
        person_count += 1
        # Draw rectangle (green color, thickness 3)
        cv2.rectangle(detected_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Add label
        cv2.putText(detected_frame, f'Person {person_count}', (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    print(f"Detected {person_count} person(s)")
    
    # Save the annotated image
    output_filename = f"detected_{image_path}"
    cv2.imwrite(output_filename, detected_frame)
    print(f"Annotated image saved as: {output_filename}")
    
    # Display images
    cv2.imshow('Original Image', frame)
    cv2.imshow('Detected Persons', detected_frame)
    print("Press any key to close the windows")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Person Detection and Recognition")
    print("=" * 50)
    print("1. Capture photo from webcam and detect persons")
    print("2. Detect persons in an existing image")
    print("=" * 50)
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == '1':
        capture_and_detect_person()
    elif choice == '2':
        image_path = input("Enter the path to the image file: ").strip()
        detect_person_in_image(image_path)
    else:
        print("Invalid choice. Please run the program again.")
