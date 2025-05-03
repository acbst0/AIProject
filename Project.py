import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

# --- AYARLAR ---
CONFIDENCE_THRESHOLD = 0.5  # Nesne tespiti güven eşiği
NMS_THRESHOLD = 0.4  # Non-maximum suppression eşiği
HEATMAP_RESOLUTION = 100  # Isı haritası çözünürlüğü

# Video seçme fonksiyonu
def select_video():
    videos_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
    
    # videos klasörü içindeki videoları listele
    video_files = [f for f in os.listdir(videos_folder) if f.endswith((".mp4", ".avi", ".mov"))]
    
    if not video_files:
        print("Videolar klasöründe video bulunamadı!")
        exit()
    
    # Seçim için gelişmiş bir tkinter penceresi oluştur
    root = tk.Tk()
    root.title("Video Seçimi")
    root.geometry("800x600")  # Pencere boyutu büyütüldü (600x400 -> 800x600)
    root.configure(bg="#f0f0f0")
    
    # Ekran ortasına yerleştir
    root.eval('tk::PlaceWindow . center')
    
    # Stil tanımla
    style = ttk.Style()
    style.theme_use('clam')  # 'clam', 'alt', 'default', 'classic' gibi temalar kullanılabilir
    
    # Başlık çerçevesi
    header_frame = tk.Frame(root, bg="#3498db", padx=10, pady=10)
    header_frame.pack(fill=tk.X)
    
    header_label = tk.Label(
        header_frame, 
        text="İşlem Yapılacak Videoyu Seçin", 
        font=("Arial", 16, "bold"),
        fg="white",
        bg="#3498db"
    )
    header_label.pack(pady=5)
    
    # Ana içerik çerçevesi
    content_frame = tk.Frame(root, bg="#f0f0f0", padx=20, pady=20)
    content_frame.pack(fill=tk.BOTH, expand=True)
    
    # Video değişkeni
    selected_video = tk.StringVar()
    
    # Video önizleme çerçevesi - BOYUTU DAHA DA BÜYÜTÜLDÜ
    preview_frame = tk.Frame(content_frame, bg="#f0f0f0", width=500, height=400)
    preview_frame.pack(side=tk.RIGHT, padx=10, pady=10)
    
    preview_label = tk.Label(
        preview_frame, 
        text="Önizleme", 
        bg="#f0f0f0", 
        font=("Arial", 14, "bold")
    )
    preview_label.pack(pady=(0, 5))
    
    # Önizleme için çok daha büyük bir alan
    thumbnail_label = tk.Label(
        preview_frame, 
        bg="#e0e0e0", 
        width=60,  # Genişlik daha fazla artırıldı
        height=25  # Yükseklik daha fazla artırıldı
    )
    thumbnail_label.pack(pady=5)
    
    # Liste çerçevesi
    list_frame = tk.Frame(content_frame, bg="#f0f0f0")
    list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    list_label = tk.Label(list_frame, text="Mevcut Videolar:", font=("Arial", 12), bg="#f0f0f0")
    list_label.pack(anchor=tk.W, pady=(0, 5))
    
    # Gelişmiş liste kutusu oluştur (Treeview)
    columns = ("name", "size", "duration")
    tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)
    
    # Sütun başlıkları
    tree.heading("name", text="Video Adı")
    tree.heading("size", text="Boyut")
    tree.heading("duration", text="Süre")
    
    # Sütun genişlikleri
    tree.column("name", width=150)
    tree.column("size", width=100, anchor=tk.CENTER)
    tree.column("duration", width=100, anchor=tk.CENTER)
    
    # Kaydırma çubuğu ekle
    scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.pack(fill=tk.BOTH, expand=True)
    
    # Önizleme oluşturma fonksiyonu
    def create_thumbnail(video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (480, 360))  # Görüntü boyutu büyük ölçüde artırıldı
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                thumbnail_label.config(image=img_tk)
                thumbnail_label.image = img_tk  # Referansı korumak için
            cap.release()
        except Exception as e:
            print(f"Önizleme oluşturulamadı: {e}")
    
    # Video bilgilerini elde et ve listeye ekle
    for video in video_files:
        video_path = os.path.join(videos_folder, video)
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB cinsinden
        
        # Video süresini almaya çalış
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
        except:
            duration = 0
        
        # Listeye ekle
        tree.insert("", tk.END, values=(video, f"{file_size:.1f} MB", f"{duration:.1f} sn"))
    
    # Seçim değiştiğinde önizleme göster
    def on_tree_select(event):
        selected_item = tree.selection()
        if selected_item:
            video_name = tree.item(selected_item[0])['values'][0]
            video_path = os.path.join(videos_folder, video_name)
            create_thumbnail(video_path)
    
    tree.bind("<<TreeviewSelect>>", on_tree_select)
    
    # Seçim fonksiyonu
    def on_select():
        selected_item = tree.selection()
        if selected_item:
            video_name = tree.item(selected_item[0])['values'][0]
            selected_video.set(os.path.join(videos_folder, video_name))
            root.destroy()
    
    # Buton çerçevesi
    button_frame = tk.Frame(root, bg="#f0f0f0", padx=20, pady=15)
    button_frame.pack(fill=tk.X)
    
    # Butonlar
    cancel_button = tk.Button(
        button_frame, 
        text="İptal", 
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
        text="Seç", 
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
    
    # İlk videoyu seç ve önizleme oluştur (varsa)
    if video_files:
        tree.selection_set(tree.get_children()[0])
        video_path = os.path.join(videos_folder, video_files[0])
        create_thumbnail(video_path)
    
    root.mainloop()
    
    # Eğer bir seçim yapılmadıysa
    if not selected_video.get():
        print("Video seçilmedi!")
        exit()
    
    return selected_video.get()

# Kullanıcıya videoyu seçtir
selected_video_path = select_video()
print(f"Seçilen video: {selected_video_path}")

# Video dosyasını aç ve ilk kareyi al
cap = cv2.VideoCapture(selected_video_path)
ret, frame = cap.read()

# Video özellikleri
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video boyutu: {frame_width}x{frame_height}, Toplam kare: {total_frames}")

if not ret:
    print("Video açılamadı veya ilk kare alınamadı.")
    exit()

# Quadrilateral (dörtgen) seçim kodu
quadrilaterals = []
current_quad = []

fig2, ax2 = plt.subplots()
ax2.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("Alan Seçimi: Her 4 tıklama bir dörtgen.\nBitirmek için ENTER'a basın.")

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

print("\nSeçilen Dörtgenler:")
for i, quad in enumerate(quadrilaterals, start=1):
    print(f"Dörtgen {i}: {quad}")

# --- KUŞBAKIŞI HARİTA ÇİZİMİ ---
print("Kuşbakışı haritayı oluşturuyor...")

# Basit bir harita oluştur - dönüşüm uygulamadan
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor('white')

# Görüntü boyutunda bir sınır belirle
min_x, max_x = 0, frame_width
min_y, max_y = 0, frame_height

# Sınırları ayarla
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)

# Izgara çiz
ax.grid(True, linestyle='-', alpha=0.7, color='#cccccc')
ax.set_axisbelow(True)

# Dörtgenleri çiz - HAM PİKSEL KOORDİNATLARINDA (Dönüşüm UYGULAMADAN)
for i, quad in enumerate(quadrilaterals):
    # Poligonu kapatmak için ilk noktayı sonuna ekle
    closed_quad = quad + [quad[0]]
    xs, ys = zip(*closed_quad)
    
    # Kenarlıkları ile birlikte çiz
    ax.fill(xs, ys, facecolor='skyblue', alpha=0.5, edgecolor='blue', linewidth=2)
    
    # Numara ekle
    cx = sum(p[0] for p in quad) / 4
    cy = sum(p[1] for p in quad) / 4
    ax.text(cx, cy, f"{i+1}", fontsize=12, color='black', 
            fontweight='bold', ha='center', va='center')

# Eksen etiketlerini göster
ax.set_xlabel('X (piksel)', fontsize=12)
ax.set_ylabel('Y (piksel)', fontsize=12)
ax.set_title("Ham Piksel Koordinatlarında Harita Görünümü", fontsize=14)

# Y eksenini ters çevir (görüntü koordinat sistemine uygun)
ax.invert_yaxis()

# Kaydet
plt.savefig("map.jpg", dpi=300, bbox_inches='tight')
plt.show()

# ========= YENİ KOD: İNSAN TESPİTİ VE ISI HARİTASI =========

# YOLO için gerekli dosyaların yolları
yolo_path = os.path.join('old', 'yolov3.cfg')
coco_names_path = os.path.join('old', 'coco.names')
weights_path = os.path.join('old', 'yolov3.weights')

# YOLO modelini yükle
print("YOLO modelini yüklüyor...")
try:
    # COCO isimlerini yükle
    with open(coco_names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # YOLOv3 ağırlıkları indir (eğer yoksa)
    if not os.path.exists(weights_path):
        print("YOLOv3 ağırlıkları indiriliyor...")
        import urllib.request
        
        url = 'https://pjreddie.com/media/files/yolov3.weights'
        print(f"İndiriliyor: {url}")
        try:
            # Ağırlık dosyasını indir
            urllib.request.urlretrieve(url, weights_path)
            print(f"İndirme tamamlandı: {weights_path}")
        except Exception as e:
            print(f"İndirme başarısız: {e}")
            weights_path = None
    
    # YOLO modelini oluştur ve CPU kullan
    if os.path.exists(weights_path):
        print(f"YOLO ağırlıkları yükleniyor: {weights_path}")
        net = cv2.dnn.readNetFromDarknet(yolo_path, weights_path)
        
        # Doğrudan CPU kullan, CUDA deneme
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("CPU kullanılacak.")
    else:
        raise FileNotFoundError("YOLO ağırlık dosyası bulunamadı")
except Exception as e:
    print(f"YOLO modeli yüklenemedi: {e}")
    print("Alternatif bir model kullanılacak...")
    
    # OpenCV'nin gömülü bir modelini kullanarak insan tespiti yap
    try:
        print("HOG tabanlı insan tespiti kullanılıyor...")
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        use_hog = True
        print("HOG modeli başarıyla yüklendi.")
    except Exception as e:
        print(f"HOG modelide yüklenemedi: {e}")
        print("İnsan tespiti yapılamayacak.")
        use_hog = False

# İnsanları tespit etmek için fonksiyon
def detect_people(frame, confidence_threshold=0.5):
    """
    Verilen frame'de insanları tespit eder
    """
    # Frame'in boyutlarını al
    height, width = frame.shape[:2]
    
    try:
        if 'use_hog' in globals() and use_hog:
            # HOG tabanlı tespit kullan
            boxes, weights = hog.detectMultiScale(
                frame, 
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05
            )
            
            person_centers = []
            for (x, y, w, h) in boxes:
                if weights.flatten()[boxes.tolist().index([x, y, w, h])] > confidence_threshold:
                    # Kutunun alt orta noktası (ayakların olduğu yer)
                    foot_x = x + w // 2
                    foot_y = y + h
                    person_centers.append((foot_x, foot_y))
            
            return person_centers
        else:
            # YOLO tabanlı tespit kullan
            # YOLO için blob oluştur
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            
            # Forward pass ve çıktıları al
            output_layers = net.getUnconnectedOutLayersNames()
            layer_outputs = net.forward(output_layers)
            
            # Tespit edilen insanların merkez noktaları
            person_centers = []
            
            # Ayak noktalarını bulmak için kutuların alt orta noktalarını al
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Sadece insanları tespit et (COCO'da 0. sınıf 'person')
                    if class_id == 0 and confidence > confidence_threshold:
                        # Tespit edilen kutu koordinatları
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Kutu koordinatları
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        # Kutunun alt orta noktası (ayakların olduğu yer)
                        foot_x = center_x
                        foot_y = y + h
                        
                        person_centers.append((foot_x, foot_y))
            
            return person_centers
    except Exception as e:
        print(f"İnsan tespiti sırasında hata: {e}")
        return []

# Isı haritası için veri toplama
print("\nİnsanları tespit ediyor ve ısı haritası için konumları kaydediyor...")
person_positions_original = []  # Orijinal piksel koordinatlarında insan konumları
person_positions_3d = []  # 3D koordinatlara dönüştürülmüş insan konumları
detection_count = 0
processed_frames = 0

# Video dosyasını yeniden aç
cap = cv2.VideoCapture(selected_video_path)
ret, first_frame = cap.read()
if not ret:
    print("Video açılamadı")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval = max(1, total_frames // 100)  # 100 kare işleyeceğiz
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Video kareleri üzerinde insan tespiti
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frames += 1
    if processed_frames % frame_interval != 0:
        continue
    
    # İnsanları tespit et
    people = detect_people(frame, CONFIDENCE_THRESHOLD)
    
    # Her insanı kaydet (HİÇBİR FİLTRELEME YAPMA)
    for person_pos in people:
        # Orijinal piksel koordinatlarını kaydet
        person_positions_original.append(person_pos)
        
        detection_count += 1
    
    if processed_frames % 10 == 0:
        print(f"İşlenen kare: {processed_frames}/{total_frames}, Tespit edilen insan: {detection_count}")

cap.release()
print(f"Toplam {processed_frames} kare işlendi, {detection_count} insan tespiti yapıldı.")

# BASİTLEŞTİRİLMİŞ HARİTA OLUŞTURMA
# Seçtiğiniz alanları ada gibi gösterip insanları doğrudan noktalar halinde gösterir
if len(person_positions_original) > 0:
    print("\nBasit haritayı oluşturuyor...")
    
    # YÖNTEM 1: BASİT HARİTA - Dörtgenler olduğu gibi
    # Basit bir beyaz arka plan oluştur
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('white')
    
    # Görüntü boyutlarında bir sınır belirle
    min_x, max_x = 0, frame_width
    min_y, max_y = 0, frame_height
    
    # Sınırları ayarla
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    # Grid çiz
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Seçilen alanları çiz
    for i, quad in enumerate(quadrilaterals):
        # Kapalı bir poligon oluştur
        closed_quad = quad + [quad[0]]
        xs, ys = zip(*closed_quad)
        
        # Alanı doldur ve kenarlarını çiz
        ax.fill(xs, ys, facecolor='skyblue', alpha=0.5, edgecolor='blue', linewidth=2)
        
        # Alan numarası ekle
        cx = sum(x for x, _ in quad) / 4
        cy = sum(y for _, y in quad) / 4
        ax.text(cx, cy, f"{i+1}", fontsize=14, color='black', fontweight='bold')
    
    # İnsan konumlarını nokta olarak çiz
    xs = [p[0] for p in person_positions_original]
    ys = [p[1] for p in person_positions_original]
    ax.scatter(xs, ys, c='red', s=20, alpha=0.5, edgecolors='none')
    
    # Başlık ve etiketler
    ax.set_title("İnsan Konumları ve Seçilen Alanlar (Filtreleme YOK)", fontsize=16)
    ax.set_xlabel("X (piksel)", fontsize=12)
    ax.set_ylabel("Y (piksel)", fontsize=12)
    
    # Y eksenini ters çevir (görüntü koordinat sistemi)
    ax.invert_yaxis()
    
    # Kaydet ve göster
    plt.savefig("map_basic.jpg", dpi=300, bbox_inches='tight')
    plt.show()
    
    # YÖNTEM 2: ISI HARİTASI - Piksel koordinatlarında - YÖN DÜZELTME
    print("Isı haritası oluşturuyor...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('white')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    # Isı haritası hesapla
    heatmap_size = HEATMAP_RESOLUTION
    x_bins = np.linspace(min_x, max_x, heatmap_size)
    y_bins = np.linspace(min_y, max_y, heatmap_size)
    
    # Histogram oluştur
    heatmap, _, _ = np.histogram2d(
        [p[0] for p in person_positions_original],
        [p[1] for p in person_positions_original],
        bins=[x_bins, y_bins]
    )
    
    # Yumuşatma uygula
    try:
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap.astype(np.float32), sigma=2)
    except Exception as e:
        print(f"Yumuşatma hatası: {e}")
    
    # Isı haritası renklerini oluştur
    colors = [(0, 0, 1, 0), (0, 0, 1, 0.3), (0, 1, 1, 0.6), (1, 1, 0, 0.8), (1, 0, 0, 1)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    
    # Isı haritasını göster - YÖN DÜZELTME
    heat = ax.imshow(
        heatmap.T,
        origin='lower',  # Diğer haritalarla aynı yönde olması için 'upper' yerine 'lower' kullan
        extent=[min_x, max_x, min_y, max_y],  # Yönü düzelt
        alpha=0.7,
        cmap=cmap,
        aspect='auto'
    )
    
    # Y eksenini ters çevir - diğer haritalarla uyumlu olması için
    ax.invert_yaxis()
    
    # Seçilen alanları çiz
    for i, quad in enumerate(quadrilaterals):
        closed_quad = quad + [quad[0]]
        xs, ys = zip(*closed_quad)
        
        # Çizgi olarak çiz, dolgu yapma (ısı haritası görünsün)
        ax.plot(xs, ys, 'blue', linewidth=2)
        
        # Alan numarası ekle
        cx = sum(x for x, _ in quad) / 4
        cy = sum(y for _, y in quad) / 4
        ax.text(cx, cy, f"{i+1}", fontsize=14, color='black', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Renk çubuğu ekle
    plt.colorbar(heat, label="İnsan yoğunluğu")
    
    # Başlık ve etiketler
    ax.set_title("İnsan Yoğunluğu Isı Haritası (Piksel koordinatlarında)", fontsize=16)
    ax.set_xlabel("X (piksel)", fontsize=12)
    ax.set_ylabel("Y (piksel)", fontsize=12)
    
    # Kaydet ve göster
    plt.savefig("map_rectified.jpg", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Orijinal video görüntüsü üzerinde sadece insan konumlarını göster (dörtgenler olmadan)
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    
    # Tespit edilen tüm insanları göster
    plt.scatter(
        [p[0] for p in person_positions_original], 
        [p[1] for p in person_positions_original],
        c='red', 
        alpha=0.5,
        s=10
    )
    
    # Başlık değiştirildi
    plt.title("Orijinal Görüntü Üzerinde İnsan Tespitleri")
    plt.savefig("map_with_detections.jpg", dpi=300)
    plt.show()
    
    print(f"İşlem tamamlandı. Haritalar kaydedildi.")
else:
    print("Hiç insan tespiti yapılamadı, haritalar oluşturulamadı.")
