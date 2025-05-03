import cv2
import numpy as np
import os
import urllib.request

def load_detection_model():
    """
    YOLO veya HOG tabanlı insan tespiti modelini yükler
    """
    yolo_path = os.path.join('old', 'yolov3.cfg')
    coco_names_path = os.path.join('old', 'coco.names')
    weights_path = os.path.join('old', 'yolov3.weights')

    print("Loading YOLO model...")
    try:
        # Load COCO names
        with open(coco_names_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        # Download YOLOv3 weights (if not exists)
        if not os.path.exists(weights_path):
            print("Downloading YOLOv3 weights...")
            
            url = 'https://pjreddie.com/media/files/yolov3.weights'
            print(f"Downloading: {url}")
            try:
                # Download weights file
                urllib.request.urlretrieve(url, weights_path)
                print(f"Download completed: {weights_path}")
            except Exception as e:
                print(f"Download failed: {e}")
                weights_path = None
        
        # Create YOLO model and use CPU
        if os.path.exists(weights_path):
            print(f"Loading YOLO weights: {weights_path}")
            net = cv2.dnn.readNetFromDarknet(yolo_path, weights_path)
            
            # Use CPU directly, don't try CUDA
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU.")
            return net, False
        else:
            raise FileNotFoundError("YOLO weights file not found")
    except Exception as e:
        print(f"Could not load YOLO model: {e}")
        print("Using an alternative model...")
        
        # Use OpenCV's built-in model for people detection
        try:
            print("Using HOG-based people detection...")
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("HOG model loaded successfully.")
            return hog, True
        except Exception as e:
            print(f"HOG model could not be loaded: {e}")
            print("People detection will not be available.")
            return None, False

def detect_people(frame, confidence_threshold=0.5, net=None, hog=None):
    """
    Frame içindeki insanları tespit eden fonksiyon
    net: YOLO modeli
    hog: HOG modeli 
    """
    height, width = frame.shape[:2]
    
    try:
        if hog is not None:
            # Use HOG-based detection
            boxes, weights = hog.detectMultiScale(
                frame, 
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05
            )
            
            person_centers = []
            for (x, y, w, h) in boxes:
                if weights.flatten()[boxes.tolist().index([x, y, w, h])] > confidence_threshold:
                    # Bottom center point of the box (where feet are)
                    foot_x = x + w // 2
                    foot_y = y + h
                    person_centers.append((foot_x, foot_y))
            
            return person_centers
        elif net is not None:
            # Use YOLO-based detection
            # Create blob for YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            
            # Forward pass and get outputs
            output_layers = net.getUnconnectedOutLayersNames()
            layer_outputs = net.forward(output_layers)
            
            # Center points of detected people
            person_centers = []
            
            # Get foot points by taking bottom middle of boxes
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Only detect people (class 0 is 'person' in COCO)
                    if class_id == 0 and confidence > confidence_threshold:
                        # Detected box coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Box coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        # Bottom center point of the box (where feet are)
                        foot_x = center_x
                        foot_y = y + h
                        
                        person_centers.append((foot_x, foot_y))
            
            return person_centers
        else:
            print("No detection model provided.")
            return []
    except Exception as e:
        print(f"Error during people detection: {e}")
        return []