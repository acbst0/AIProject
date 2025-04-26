import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

# --- AYARLAR ---
CONFIDENCE_THRESHOLD = 0.5  # Nesne tespiti güven eşiği
NMS_THRESHOLD = 0.4  # Non-maximum suppression eşiği
HEATMAP_RESOLUTION = 100  # Isı haritası çözünürlüğü

# Video dosyasını aç ve ilk kareyi al
cap = cv2.VideoCapture('camera.mp4')
ret, frame = cap.read()

# Video özellikleri
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video boyutu: {frame_width}x{frame_height}, Toplam kare: {total_frames}")

if not ret:
    print("Video açılamadı veya ilk kare alınamadı.")
    exit()

# Noktalar burada toplanacak
points = []

# Başlangıç görseli
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("Lütfen sırasıyla 4 nokta seçin:\n(Referans, X, Y, Z)")

# Tıklama fonksiyonu
def onclick(event):
    if event.xdata is not None and event.ydata is not None and len(points) < 4:
        x, y = event.xdata, event.ydata
        points.append((x, y))
        ax.plot(x, y, 'ro')  # kırmızı nokta
        ax.text(x + 5, y, f"{len(points)}", color='yellow', fontsize=12)  # numaralandır
        plt.draw()
        if len(points) == 4:
            plt.close()

# Event bağlama ve gösterme
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# Eğer 4 nokta seçilmediyse çık
if len(points) != 4:
    print("Yeterli sayıda nokta seçilmedi.")
    exit()

# Koordinat hesaplama
origin = points[0]
x_axis = (origin[0] - points[1][0], origin[1] - points[1][1])
y_axis = (origin[0] - points[2][0], origin[1] - points[2][1])
z_axis = (origin[0] - points[3][0], origin[1] - points[3][1])

""" x_axis = (points[1][0] - origin[0], points[1][1] - origin[1])
y_axis = (points[2][0] - origin[0], points[2][1] - origin[1])
z_axis = (points[3][0] - origin[0], points[3][1] - origin[1]) """

# kordinatları orijine göre uzatarak fotoğrafın eksenleri ile hizalama

print("\nKoordinat Sistemi:")
print(f"Origin (0,0,0): {origin}")
print(f"X Axis (1,0,0): {x_axis}")
print(f"Y Axis (0,1,0): {y_axis}")
print(f"Z Axis (0,0,1): {z_axis}")

# --- Eksenleri görselin sınırlarına kadar uzatma fonksiyonu ---
def extend_axis_to_bounds(origin, direction, image_shape, scale_factor=1000):
    """
    origin: (x0, y0) başlangıç noktası
    direction: (dx, dy) yön vektörü
    image_shape: (yükseklik, genişlik, kanal)
    """
    x0, y0 = origin
    dx, dy = direction
    norm = np.sqrt(dx**2 + dy**2)
    if norm == 0:
        return origin, origin  # Sabit nokta, vektör yok

    # Birim vektör
    dx /= norm
    dy /= norm

    # Uzaklaştırılmış iki nokta
    pt1 = (x0 - dx * scale_factor, y0 - dy * scale_factor)
    pt2 = (x0 + dx * scale_factor, y0 + dy * scale_factor)

    return pt1, pt2

# Görselde eksenleri çizme
fig2, ax2 = plt.subplots()
ax2.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("Eksenlerin Uzatılması")

# X ekseni uzatılması
pt1, pt2 = extend_axis_to_bounds(origin, x_axis, frame.shape)
ax2.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', label='X Ekseni')

# Y ekseni uzatılması
pt1_y, pt2_y = extend_axis_to_bounds(origin, y_axis, frame.shape)
ax2.plot([pt1_y[0], pt2_y[0]], [pt1_y[1], pt2_y[1]], 'g-', label='Y Ekseni')

# Z ekseni uzatılması
pt1_z, pt2_z = extend_axis_to_bounds(origin, z_axis, frame.shape)
ax2.plot([pt1_z[0], pt2_z[0]], [pt1_z[1], pt2_z[1]], 'b-', label='Z Ekseni')

# Orijin noktası
ax2.plot(origin[0], origin[1], 'yo', markersize=8)
ax2.text(origin[0]+5, origin[1], "Origin", color='yellow', fontsize=10)

ax2.legend()
plt.show()

# Quadrilateral (dörtgen) seçim kodu devam eder
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

# --- Koordinat dönüşüm işlemleri ---
def compute_3d_basis(origin, x_axis, y_axis, z_axis):
    x_vec = np.array(x_axis)
    y_vec = np.array(y_axis)
    z_vec = np.array(z_axis)
    origin_vec = np.array(origin)
    
    # Birim vektörlere dönüştür
    x_unit = x_vec / np.linalg.norm(x_vec)
    y_unit = y_vec / np.linalg.norm(y_vec)
    z_unit = z_vec / np.linalg.norm(z_vec)
    
    # Ortogonal bir baz oluştur
    B = np.vstack([x_unit, y_unit, z_unit]).T
    return B, origin_vec

def resolve_to_3d(p, B, origin_vec):
    vec = np.array(p) - origin_vec
    coords_3d, _, _, _ = np.linalg.lstsq(B, vec, rcond=None)
    return coords_3d

B, origin_vec = compute_3d_basis(origin, x_axis, y_axis, z_axis)

# --- YENİ DÜZGÜN PERSPEKTİF DÖNÜŞÜMÜ ---
def create_top_view_transformation(quads_2d):
    """
    Daha gelişmiş bir perspektif dönüşümü için homografi matrisini hesaplar.
    Paralel çizgilerin paralel kalmasını sağlayan bir dönüşüm.
    """
    # Birinci dörtgeni kullanarak ana yönleri belirle
    if len(quads_2d) > 0:
        quad = quads_2d[0]  # İlk dörtgeni referans al
        
        # Dörtgenin merkezini bul
        cx = sum(p[0] for p in quad) / 4
        cy = sum(p[1] for p in quad) / 4
        
        # Merkeze göre vektörleri hesapla
        vectors = []
        for p in quad:
            vec = np.array([p[0] - cx, p[1] - cy])
            vectors.append(vec)
        
        # İki ana yönü bul (birbirine en uzak köşeleri kullan)
        main_dir1 = None
        main_dir2 = None
        max_dist = 0
        
        for i in range(4):
            for j in range(i + 1, 4):
                dist = np.linalg.norm(vectors[i] - vectors[j])
                if dist > max_dist:
                    max_dist = dist
                    main_dir1 = vectors[i]
                    main_dir2 = vectors[j]
        
        # X ve Y eksenlerini belirle
        x_dir = main_dir1 / np.linalg.norm(main_dir1)
        y_dir = main_dir2 / np.linalg.norm(main_dir2)
        
        # Ortogonallik kontrolü ve düzeltme
        dot_product = np.dot(x_dir, y_dir)
        if abs(dot_product) > 0.1:  # Eksenler yeteri kadar dik değilse
            # Y eksenini X'e dik olacak şekilde düzelt
            y_dir = y_dir - dot_product * x_dir
            y_dir = y_dir / np.linalg.norm(y_dir)
    else:
        # Varsayılan eksenler
        x_dir = np.array([1, 0])
        y_dir = np.array([0, 1])
    
    # Standart dönüşüm matrisi oluştur
    return x_dir, y_dir

# Dönüşüm için 2D koordinatları hesapla
bird_eye_quads = []
for quad in quadrilaterals:
    bird_eye_quad = []
    for p in quad:
        x, y, z = resolve_to_3d(p, B, origin_vec)
        bird_eye_quad.append((x, y))
    bird_eye_quads.append(bird_eye_quad)

# Paralel çizgileri koruyacak dönüşüm eksenlerini hesapla
x_dir, y_dir = create_top_view_transformation(bird_eye_quads)

# Her dörtgeni düzelt - paralel çizgileri koruyarak
print("\nDörtgenlerin kuş bakışı görünümünü oluşturuyor...")
corrected_quads = []
for quad in bird_eye_quads:
    corrected_quad = []
    for p in quad:
        # Koordinatları x_dir ve y_dir eksenlerine göre düzelt
        x_new = p[0]  # X koordinatı olduğu gibi bırak
        y_new = p[1]  # Y koordinatı olduğu gibi bırak
        corrected_quad.append((x_new, y_new))
    corrected_quads.append(corrected_quad)

# --- TEMEL KUŞBAKIŞI PROJEKSİYON ---
def create_birds_eye_view(quads, width=800, height=800, margin=50):
    """
    Kuş bakışı görünümü oluştur - input olarak 2D koordinattaki dörtgenleri alır
    """
    # Tüm noktaları tek bir listede topla
    all_points = []
    for quad in quads:
        # Z koordinatını çıkar (varsa)
        quad_2d = [(p[0], p[1]) if len(p) == 2 else (p[0], p[1]) for p in quad]
        all_points.extend(quad_2d)
    
    # Min/max değerleri hesapla
    all_points = np.array(all_points)
    x_min, y_min = np.min(all_points[:, 0]), np.min(all_points[:, 1])
    x_max, y_max = np.max(all_points[:, 0]), np.max(all_points[:, 1])
    
    # Ölçek hesapla
    x_scale = (width - 2 * margin) / (x_max - x_min)
    y_scale = (height - 2 * margin) / (y_max - y_min)
    scale = min(x_scale, y_scale)
    
    # Merkezi hizala
    x_offset = ((width - ((x_max - x_min) * scale)) / 2) + margin - x_min * scale
    y_offset = ((height - ((y_max - y_min) * scale)) / 2) + margin - y_min * scale
    
    # Dörtgenleri ölçekle ve hizala
    transformed_quads = []
    for quad in quads:
        transformed_quad = []
        for p in quad:
            # Z koordinatını çıkar (varsa)
            x, y = p[0], p[1]
            x_new = x * scale + x_offset
            y_new = y * scale + y_offset
            transformed_quad.append((x_new, y_new))
        transformed_quads.append(transformed_quad)
    
    return transformed_quads, (x_offset, y_offset, scale)

# Düzeltilmiş kuş bakışı koordinatlarını kullan
transformed_quads, transform_params = create_birds_eye_view(corrected_quads)

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

# Orijinal görüntü üzerine izdüşüm gösterimini kaldırıyoruz - kullanıcının isteği üzerine

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
cap = cv2.VideoCapture('camera.mp4')
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
        
        # Sadece bilgi amaçlı 3D koordinatları da hesapla
        try:
            person_3d = resolve_to_3d(person_pos, B, origin_vec)
            person_positions_3d.append((person_3d[0], person_3d[1]))
        except:
            # Hata olursa sadece devam et, hesaplanamayan koordinatlar atlanır
            pass
        
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
