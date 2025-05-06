import cv2
import numpy as np
import os
import tkinter as tk
from video_selector import select_video
from quadrilateral_selector import select_quadrilaterals
from people_detector import load_detection_model, detect_people
from map_generator import create_basic_map, create_heatmap, create_detections_on_image, create_category_analysis

# İnsan tespiti için kullanılacak minimum güven skoru
CONFIDENCE_THRESHOLD = 0.5
# YOLO için Non-Maximum Suppression eşiği (şu an kullanılmıyor ama modelde gerekirse kullanılabilir)
NMS_THRESHOLD = 0.4
# Isı haritası çözünürlüğü (daha yüksek değer, daha detaylı ısı haritası demek)
HEATMAP_RESOLUTION = 30


def main():
    # 1. Kullanıcıdan işlenecek videoyu seçmesini iste
    selected_video_path = select_video()
    print(f"Selected video: {selected_video_path}")

    # 2. Videoyu aç ve ilk frame'i al
    cap = cv2.VideoCapture(selected_video_path)
    ret, frame = cap.read()

    # 3. Video özelliklerini al (genişlik, yükseklik, toplam frame sayısı)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video size: {frame_width}x{frame_height}, Total frames: {total_frames}")

    if not ret:
        print("Could not open video or read first frame.")
        exit()

    # 4. İlk frame üzerinde kullanıcıdan dörtgen alan(lar) seçmesini iste
    quadrilaterals, categories = select_quadrilaterals(frame)
    print("\nSelected Quadrilaterals and Categories:")
    for i, (quad, category) in enumerate(zip(quadrilaterals, categories), start=1):
        print(f"Quadrilateral {i} ({category}): {quad}")

    # 5. Seçilen alanlarla temel harita oluştur
    create_basic_map(frame_width, frame_height, quadrilaterals, categories)

    # 6. İnsan tespiti için modeli yükle (önce YOLO, olmazsa HOG)
    detection_model, use_hog = load_detection_model()

    # 7. Video boyunca insan tespiti yapıp, tespit edilen kişilerin konumlarını kaydet
    print("\nDetecting people and recording locations for heatmap...")
    person_positions_original = []  # Tespit edilen kişilerin (x, y) konumları
    person_categories = []  # Her kişinin ait olduğu kategori
    detection_count = 0
    processed_frames = 0

    # Videoyu baştan aç (frame atlamak için)
    cap = cv2.VideoCapture(selected_video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Could not open video")
        exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 100 frame işleyecek şekilde aralık belirle (daha hızlı çalışması için)
    frame_interval = max(1, total_frames // 100)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 8. Video boyunca frame'leri işle
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frames += 1
        # Sadece belirli aralıktaki frame'leri işle
        if processed_frames % frame_interval != 0:
            continue
        # İnsan tespiti yap
        if use_hog:
            people = detect_people(frame, CONFIDENCE_THRESHOLD, hog=detection_model)
        else:
            people = detect_people(frame, CONFIDENCE_THRESHOLD, net=detection_model)
        # Tespit edilen her kişiyi kaydet
        for person_pos in people:
            person_positions_original.append(person_pos)
            
            # Kişinin hangi kategoriye ait olduğunu belirle
            category_id = find_person_category(person_pos, quadrilaterals)
            if category_id != -1:
                person_categories.append(categories[category_id])
            else:
                person_categories.append("Outside")
                
            detection_count += 1
        # Her 10 frame'de bir ilerleme yazdır
        if processed_frames % 10 == 0:
            print(f"Processed frames: {processed_frames}/{total_frames}, Detected people: {detection_count}")

    cap.release()
    print(f"Processed a total of {processed_frames} frames, detected {detection_count} people.")

    # 9. Eğer insan tespiti yapıldıysa, haritaları oluştur
    if len(person_positions_original) > 0:
        # a) İnsanların konumlarını gösteren basit harita
        create_heatmap(frame_width, frame_height, quadrilaterals, person_positions_original, 
                      HEATMAP_RESOLUTION, "People Locations and Selected Areas (NO FILTERING)",
                      "map_basic.jpg", categories=categories)
        # b) Yoğunluk (ısı) haritası
        create_heatmap(frame_width, frame_height, quadrilaterals, person_positions_original, 
                      HEATMAP_RESOLUTION, "People Density Heatmap (in pixel coordinates)",
                      "map_rectified.jpg", show_heatmap=True, categories=categories)
        # c) Orijinal görüntü üzerinde tespit edilen kişileri göster
        create_detections_on_image(first_frame, person_positions_original)
        # d) Kategori bazlı analizler
        create_category_analysis(person_positions_original, person_categories)
        print(f"Process completed. Maps saved.")
    else:
        print("No people were detected, could not create maps.")


def find_person_category(person_pos, quadrilaterals):
    """
    Kişinin hangi kategoriye ait olduğunu belirlemek için en yakın veya içinde olduğu dörtgeni bulur
    """
    x, y = person_pos
    
    # Önce kişinin içinde olduğu dörtgeni kontrol et
    for i, quad in enumerate(quadrilaterals):
        if point_in_polygon(x, y, quad):
            return i
    
    # İçinde değilse, en yakın dörtgeni bul
    min_distance = float('inf')
    closest_quad_idx = -1
    
    for i, quad in enumerate(quadrilaterals):
        distance = min_distance_to_polygon(x, y, quad)
        if distance < min_distance:
            min_distance = distance
            closest_quad_idx = i
            
    return closest_quad_idx


def point_in_polygon(x, y, polygon):
    """
    Ray casting algoritması ile bir noktanın çokgen içinde olup olmadığını belirler
    """
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def min_distance_to_polygon(x, y, polygon):
    """
    Bir noktanın çokgene olan minimum mesafesini hesaplar
    """
    n = len(polygon)
    min_dist = float('inf')
    
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        
        # Nokta ile kenar arasındaki mesafeyi hesapla
        dist = distance_to_line_segment(x, y, p1[0], p1[1], p2[0], p2[1])
        min_dist = min(min_dist, dist)
    
    return min_dist


def distance_to_line_segment(x, y, x1, y1, x2, y2):
    """
    Bir noktanın bir doğru parçasına olan mesafesini hesaplar
    """
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    
    if len_sq == 0:  # Doğru parçası bir noktaysa
        return np.sqrt(A * A + B * B)
    
    param = dot / len_sq
    
    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D
    
    return np.sqrt((x - xx) ** 2 + (y - yy) ** 2)


# Program doğrudan çalıştırıldığında ana fonksiyonu başlat
if __name__ == "__main__":
    main()