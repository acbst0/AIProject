import cv2
import numpy as np
import os
import tkinter as tk
from video_selector import select_video
from quadrilateral_selector import select_quadrilaterals
from people_detector import load_detection_model, detect_people
from map_generator import create_basic_map, create_heatmap, create_detections_on_image

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
HEATMAP_RESOLUTION = 100

def main():
    # Let the user select a video
    selected_video_path = select_video()
    print(f"Selected video: {selected_video_path}")

    # Open video file and get first frame
    cap = cv2.VideoCapture(selected_video_path)
    ret, frame = cap.read()

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video size: {frame_width}x{frame_height}, Total frames: {total_frames}")

    if not ret:
        print("Could not open video or read first frame.")
        exit()

    # Select quadrilaterals from the first frame
    quadrilaterals = select_quadrilaterals(frame)
    print("\nSelected Quadrilaterals:")
    for i, quad in enumerate(quadrilaterals, start=1):
        print(f"Quadrilateral {i}: {quad}")

    # Create basic map with quadrilaterals
    create_basic_map(frame_width, frame_height, quadrilaterals)

    # Load detection model
    detection_model, use_hog = load_detection_model()

    # Detect people in video
    print("\nDetecting people and recording locations for heatmap...")
    person_positions_original = []
    detection_count = 0
    processed_frames = 0

    # Reopen video file
    cap = cv2.VideoCapture(selected_video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Could not open video")
        exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // 100)  # We'll process 100 frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # People detection on video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frames += 1
        if processed_frames % frame_interval != 0:
            continue
        
        # Detect people
        if use_hog:
            people = detect_people(frame, CONFIDENCE_THRESHOLD, hog=detection_model)
        else:
            people = detect_people(frame, CONFIDENCE_THRESHOLD, net=detection_model)
        
        # Save each person (NO FILTERING)
        for person_pos in people:
            # Save original pixel coordinates
            person_positions_original.append(person_pos)
            
            detection_count += 1
        
        if processed_frames % 10 == 0:
            print(f"Processed frames: {processed_frames}/{total_frames}, Detected people: {detection_count}")

    cap.release()
    print(f"Processed a total of {processed_frames} frames, detected {detection_count} people.")

    # Create maps if people were detected
    if len(person_positions_original) > 0:
        # Create simple map with people
        create_heatmap(frame_width, frame_height, quadrilaterals, person_positions_original, 
                      HEATMAP_RESOLUTION, "People Locations and Selected Areas (NO FILTERING)",
                      "map_basic.jpg")
        
        # Create heatmap
        create_heatmap(frame_width, frame_height, quadrilaterals, person_positions_original, 
                      HEATMAP_RESOLUTION, "People Density Heatmap (in pixel coordinates)",
                      "map_rectified.jpg", show_heatmap=True)
        
        # Show detections on original image
        create_detections_on_image(first_frame, person_positions_original)
        
        print(f"Process completed. Maps saved.")
    else:
        print("No people were detected, could not create maps.")

if __name__ == "__main__":
    main()