import cv2
import numpy as np
import os
import urllib.request

# YOLO veya HOG tabanlı insan tespiti modelini yükler.
# Önce YOLO modelini yüklemeye çalışır, başarısız olursa HOG modeline geçer.
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

# Frame içindeki insanları tespit eden fonksiyon.
# YOLO veya HOG modeline göre tespit yapar ve kişilerin ayak noktalarını (x, y) olarak döndürür.
def detect_people(frame, confidence_threshold=0.5, net=None, hog=None):
    """
    Frame içindeki insanları tespit eden fonksiyon
    net: YOLO modeli
    hog: HOG modeli 
    """
    height, width = frame.shape[:2]
    
    try:
        if hog is not None:
            # HOG tabanlı insan tespiti yapılır.
            # detectMultiScale fonksiyonu, görüntüdeki insanları dikdörtgenler (x, y, w, h) olarak döndürür.
            # weights: Her tespit için güven skoru içerir.
            boxes, weights = hog.detectMultiScale(
                frame, 
                winStride=(8, 8),  # Arama penceresinin kayma adımı (piksel cinsinden)
                padding=(4, 4),    # Kenar boşluğu (piksel cinsinden)
                scale=1.05         # Görüntü ölçeklendirme oranı (daha fazla tespit için 1.01-1.1 arası olabilir)
            )
            
            person_centers = []
            for idx, (x, y, w, h) in enumerate(boxes):
                # Her tespit edilen dikdörtgen için güven skoru kontrol edilir
                if weights.flatten()[idx] > confidence_threshold:
                    # Kişinin ayak ucu koordinatı (x ekseninde ortası, y ekseninde altı) alınır
                    foot_x = x + w // 2
                    foot_y = y + h
                    person_centers.append((foot_x, foot_y))
            # Tespit edilen kişilerin ayak ucu koordinatları döndürülür
            return person_centers
        elif net is not None:
            # YOLO tabanlı insan tespiti yapılır.
            # Görüntüden blob (YOLO için uygun giriş) oluşturulur
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            
            # Modelin çıkış katmanları alınır ve ileri yayılım yapılır
            output_layers = net.getUnconnectedOutLayersNames()
            layer_outputs = net.forward(output_layers)
            
            person_centers = []
            # Her tespit edilen obje için skorlar ve sınıf kimliği kontrol edilir
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]  # Sınıf skorları
                    class_id = np.argmax(scores)  # En yüksek skora sahip sınıf
                    confidence = scores[class_id]  # O sınıfın güven skoru
                    # Sadece insan sınıfı (COCO veri setinde class_id=0) ve güven eşiği kontrolü
                    if class_id == 0 and confidence > confidence_threshold:
                        # Tespit edilen kutunun merkez koordinatları ve boyutları
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        # Kişinin ayak ucu koordinatı (alt orta nokta)
                        foot_x = center_x
                        foot_y = y + h
                        person_centers.append((foot_x, foot_y))
            # Tespit edilen kişilerin ayak ucu koordinatları döndürülür
            return person_centers
        else:
            # Model yüklenemediyse boş liste döndürülür
            print("Tespit modeli bulunamadı.")
            return []
    except Exception as e:
        # Hata oluşursa ekrana yazdırılır ve boş liste döndürülür
        print(f"İnsan tespiti sırasında hata oluştu: {e}")
        return []