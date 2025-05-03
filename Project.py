import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
HEATMAP_RESOLUTION = 100

def select_video():
    videos_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
    
    video_files = [f for f in os.listdir(videos_folder) if f.endswith((".mp4", ".avi", ".mov"))]
    
    if not video_files:
        print("No videos found in the videos folder!")
        exit()
    
    root = tk.Tk()
    root.title("Video Selection")
    root.geometry("800x600")
    root.configure(bg="#f0f0f0")
    
    root.eval('tk::PlaceWindow . center')
    
    style = ttk.Style()
    style.theme_use('clam')
    
    header_frame = tk.Frame(root, bg="#3498db", padx=10, pady=10)
    header_frame.pack(fill=tk.X)
    
    header_label = tk.Label(
        header_frame, 
        text="Select Video to Process", 
        font=("Arial", 16, "bold"),
        fg="white",
        bg="#3498db"
    )
    header_label.pack(pady=5)
    
    content_frame = tk.Frame(root, bg="#f0f0f0", padx=20, pady=20)
    content_frame.pack(fill=tk.BOTH, expand=True)
    
    selected_video = tk.StringVar()
    
    preview_frame = tk.Frame(content_frame, bg="#f0f0f0", width=500, height=400)
    preview_frame.pack(side=tk.RIGHT, padx=10, pady=10)
    
    preview_label = tk.Label(
        preview_frame, 
        text="Preview", 
        bg="#f0f0f0", 
        font=("Arial", 14, "bold")
    )
    preview_label.pack(pady=(0, 5))
    
    thumbnail_label = tk.Label(
        preview_frame, 
        bg="#e0e0e0", 
        width=60,
        height=25
    )
    thumbnail_label.pack(pady=5)
    
    list_frame = tk.Frame(content_frame, bg="#f0f0f0")
    list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    list_label = tk.Label(list_frame, text="Available Videos:", font=("Arial", 12), bg="#f0f0f0")
    list_label.pack(anchor=tk.W, pady=(0, 5))
    
    columns = ("name", "size", "duration")
    tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)
    
    tree.heading("name", text="Video Name")
    tree.heading("size", text="Size")
    tree.heading("duration", text="Duration")
    
    tree.column("name", width=150)
    tree.column("size", width=100, anchor=tk.CENTER)
    tree.column("duration", width=100, anchor=tk.CENTER)
    
    scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.pack(fill=tk.BOTH, expand=True)
    
    def create_thumbnail(video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (480, 360))
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                thumbnail_label.config(image=img_tk)
                thumbnail_label.image = img_tk
            cap.release()
        except Exception as e:
            print(f"Could not create preview: {e}")
    
    for video in video_files:
        video_path = os.path.join(videos_folder, video)
        file_size = os.path.getsize(video_path) / (1024 * 1024)
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
        except:
            duration = 0
        
        tree.insert("", tk.END, values=(video, f"{file_size:.1f} MB", f"{duration:.1f} sec"))
    
    def on_tree_select(event):
        selected_item = tree.selection()
        if selected_item:
            video_name = tree.item(selected_item[0])['values'][0]
            video_path = os.path.join(videos_folder, video_name)
            create_thumbnail(video_path)
    
    tree.bind("<<TreeviewSelect>>", on_tree_select)
    
    def on_select():
        selected_item = tree.selection()
        if selected_item:
            video_name = tree.item(selected_item[0])['values'][0]
            selected_video.set(os.path.join(videos_folder, video_name))
            root.destroy()
    
    button_frame = tk.Frame(root, bg="#f0f0f0", padx=20, pady=15)
    button_frame.pack(fill=tk.X)
    
    cancel_button = tk.Button(
        button_frame, 
        text="Cancel", 
        command=root.destroy,
        bg="#e74c3c",
        fg="white",
        font=("Arial", 11),
        width=10,
        height=1,
        relief=tk.RAISED,
        cursor="hand2"
    )
    cancel_button.pack(side=tk.LEFT, padx=10)
    
    select_button = tk.Button(
        button_frame, 
        text="Select", 
        command=on_select,
        bg="#2ecc71",
        fg="white",
        font=("Arial", 11, "bold"),
        width=10,
        height=1,
        relief=tk.RAISED,
        cursor="hand2"
    )
    select_button.pack(side=tk.RIGHT, padx=10)
    
    if video_files:
        tree.selection_set(tree.get_children()[0])
        video_path = os.path.join(videos_folder, video_files[0])
        create_thumbnail(video_path)
    
    root.mainloop()
    
    if not selected_video.get():
        print("No video selected!")
        exit()
    
    return selected_video.get()

selected_video_path = select_video()
print(f"Selected video: {selected_video_path}")

cap = cv2.VideoCapture(selected_video_path)
ret, frame = cap.read()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video size: {frame_width}x{frame_height}, Total frames: {total_frames}")

if not ret:
    print("Could not open video or read first frame.")
    exit()

quadrilaterals = []
current_quad = []

fig2, ax2 = plt.subplots()
ax2.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("Area Selection: Every 4 clicks make a quadrilateral.\nPress ENTER when finished.")

def quad_onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        current_quad.append((x, y))
        ax2.plot(x, y, 'go')
        ax2.text(x + 5, y, f"{len(current_quad)}", color='lime', fontsize=10)
        plt.draw()

        if len(current_quad) == 4:
            xs, ys = zip(*current_quad + [current_quad[0]])
            ax2.plot(xs, ys, 'g-')
            quad_id = len(quadrilaterals) + 1
            cx = sum(x for x, _ in current_quad) / 4
            cy = sum(y for _, y in current_quad) / 4
            ax2.text(cx, cy, f"#{quad_id}", color='white', fontsize=12, weight='bold')
            quadrilaterals.append(current_quad.copy())
            current_quad.clear()
            plt.draw()

def onkey(event):
    if event.key == 'enter':
        plt.close()

fig2.canvas.mpl_connect('button_press_event', quad_onclick)
fig2.canvas.mpl_connect('key_press_event', onkey)
plt.show()

print("\nSelected Quadrilaterals:")
for i, quad in enumerate(quadrilaterals, start=1):
    print(f"Quadrilateral {i}: {quad}")

print("Creating bird's eye view map...")

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor('white')

min_x, max_x = 0, frame_width
min_y, max_y = 0, frame_height

ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)

ax.grid(True, linestyle='-', alpha=0.7, color='#cccccc')
ax.set_axisbelow(True)

for i, quad in enumerate(quadrilaterals):
    closed_quad = quad + [quad[0]]
    xs, ys = zip(*closed_quad)
    
    ax.fill(xs, ys, facecolor='skyblue', alpha=0.5, edgecolor='blue', linewidth=2)
    
    cx = sum(p[0] for p in quad) / 4
    cy = sum(p[1] for p in quad) / 4
    ax.text(cx, cy, f"{i+1}", fontsize=12, color='black', 
            fontweight='bold', ha='center', va='center')

ax.set_xlabel('X (pixels)', fontsize=12)
ax.set_ylabel('Y (pixels)', fontsize=12)
ax.set_title("Map View in Raw Pixel Coordinates", fontsize=14)

ax.invert_yaxis()

plt.savefig("map.jpg", dpi=300, bbox_inches='tight')
plt.show()

yolo_path = os.path.join('old', 'yolov3.cfg')
coco_names_path = os.path.join('old', 'coco.names')
weights_path = os.path.join('old', 'yolov3.weights')

print("Loading YOLO model...")
try:
    with open(coco_names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    if not os.path.exists(weights_path):
        print("Downloading YOLOv3 weights...")
        import urllib.request
        
        url = 'https://pjreddie.com/media/files/yolov3.weights'
        print(f"Downloading: {url}")
        try:
            urllib.request.urlretrieve(url, weights_path)
            print(f"Download completed: {weights_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            weights_path = None
    
    if os.path.exists(weights_path):
        print(f"Loading YOLO weights: {weights_path}")
        net = cv2.dnn.readNetFromDarknet(yolo_path, weights_path)
        
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU.")
    else:
        raise FileNotFoundError("YOLO weights file not found")
except Exception as e:
    print(f"Could not load YOLO model: {e}")
    print("Using an alternative model...")
    
    try:
        print("Using HOG-based people detection...")
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        use_hog = True
        print("HOG model loaded successfully.")
    except Exception as e:
        print(f"HOG model could not be loaded: {e}")
        print("People detection will not be available.")
        use_hog = False

def detect_people(frame, confidence_threshold=0.5):
    height, width = frame.shape[:2]
    
    try:
        if 'use_hog' in globals() and use_hog:
            boxes, weights = hog.detectMultiScale(
                frame, 
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05
            )
            
            person_centers = []
            for (x, y, w, h) in boxes:
                if weights.flatten()[boxes.tolist().index([x, y, w, h])] > confidence_threshold:
                    foot_x = x + w // 2
                    foot_y = y + h
                    person_centers.append((foot_x, foot_y))
            
            return person_centers
        else:
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            
            output_layers = net.getUnconnectedOutLayersNames()
            layer_outputs = net.forward(output_layers)
            
            person_centers = []
            
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if class_id == 0 and confidence > confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        foot_x = center_x
                        foot_y = y + h
                        
                        person_centers.append((foot_x, foot_y))
            
            return person_centers
    except Exception as e:
        print(f"Error during people detection: {e}")
        return []

print("\nDetecting people and recording locations for heatmap...")
person_positions_original = []
person_positions_3d = []
detection_count = 0
processed_frames = 0

cap = cv2.VideoCapture(selected_video_path)
ret, first_frame = cap.read()
if not ret:
    print("Could not open video")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval = max(1, total_frames // 100)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frames += 1
    if processed_frames % frame_interval != 0:
        continue
    
    people = detect_people(frame, CONFIDENCE_THRESHOLD)
    
    for person_pos in people:
        person_positions_original.append(person_pos)
        
        detection_count += 1
    
    if processed_frames % 10 == 0:
        print(f"Processed frames: {processed_frames}/{total_frames}, Detected people: {detection_count}")

cap.release()
print(f"Processed a total of {processed_frames} frames, detected {detection_count} people.")

if len(person_positions_original) > 0:
    print("\nCreating simple map...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('white')
    
    min_x, max_x = 0, frame_width
    min_y, max_y = 0, frame_height
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    
    for i, quad in enumerate(quadrilaterals):
        closed_quad = quad + [quad[0]]
        xs, ys = zip(*closed_quad)
        
        ax.fill(xs, ys, facecolor='skyblue', alpha=0.5, edgecolor='blue', linewidth=2)
        
        cx = sum(x for x, _ in quad) / 4
        cy = sum(y for _, y in quad) / 4
        ax.text(cx, cy, f"{i+1}", fontsize=14, color='black', fontweight='bold')
    
    xs = [p[0] for p in person_positions_original]
    ys = [p[1] for p in person_positions_original]
    ax.scatter(xs, ys, c='red', s=20, alpha=0.5, edgecolors='none')
    
    ax.set_title("People Locations and Selected Areas (NO FILTERING)", fontsize=16)
    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)
    
    ax.invert_yaxis()
    
    plt.savefig("map_basic.jpg", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Creating heatmap...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('white')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    heatmap_size = HEATMAP_RESOLUTION
    x_bins = np.linspace(min_x, max_x, heatmap_size)
    y_bins = np.linspace(min_y, max_y, heatmap_size)
    
    heatmap, _, _ = np.histogram2d(
        [p[0] for p in person_positions_original],
        [p[1] for p in person_positions_original],
        bins=[x_bins, y_bins]
    )
    
    try:
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap.astype(np.float32), sigma=2)
    except Exception as e:
        print(f"Smoothing error: {e}")
    
    colors = [(0, 0, 1, 0), (0, 0, 1, 0.3), (0, 1, 1, 0.6), (1, 1, 0, 0.8), (1, 0, 0, 1)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    
    heat = ax.imshow(
        heatmap.T,
        origin='lower',
        extent=[min_x, max_x, min_y, max_y],
        alpha=0.7,
        cmap=cmap,
        aspect='auto'
    )
    
    ax.invert_yaxis()
    
    for i, quad in enumerate(quadrilaterals):
        closed_quad = quad + [quad[0]]
        xs, ys = zip(*closed_quad)
        
        ax.plot(xs, ys, 'blue', linewidth=2)
        
        cx = sum(x for x, _ in quad) / 4
        cy = sum(y for _, y in quad) / 4
        ax.text(cx, cy, f"{i+1}", fontsize=14, color='black', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    plt.colorbar(heat, label="People density")
    
    ax.set_title("People Density Heatmap (in pixel coordinates)", fontsize=16)
    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)
    
    plt.savefig("map_rectified.jpg", dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    
    plt.scatter(
        [p[0] for p in person_positions_original], 
        [p[1] for p in person_positions_original],
        c='red', 
        alpha=0.5,
        s=10
    )
    
    plt.title("People Detections on Original Image")
    plt.savefig("map_with_detections.jpg", dpi=300)
    plt.show()
    
    print(f"Process completed. Maps saved.")
else:
    print("No people were detected, could not create maps.")
