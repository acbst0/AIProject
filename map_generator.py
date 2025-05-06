import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from collections import Counter

def ensure_maps_folder_exists():
    """
    maps klasörünün var olduğundan emin olur, yoksa oluşturur
    """
    # maps klasörünün yolu belirlenir
    maps_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")
    # Eğer klasör yoksa oluşturulur
    if not os.path.exists(maps_folder):
        os.makedirs(maps_folder)
        print(f"Created maps folder: {maps_folder}")
    return maps_folder


def create_basic_map(frame_width, frame_height, quadrilaterals, categories=None):
    """
    Temel kuşbakışı haritayı oluşturur
    """
    print("Creating bird's eye view map...")
    maps_folder = ensure_maps_folder_exists()

    # Harita için boş bir figür oluşturulur
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('white')

    # Eksen sınırları video boyutuna göre ayarlanır
    min_x, max_x = 0, frame_width
    min_y, max_y = 0, frame_height
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Izgara ve eksen ayarları
    ax.grid(True, linestyle='-', alpha=0.7, color='#cccccc')
    ax.set_axisbelow(True)

    # Seçilen her dörtgen alanı çiz ve numaralandır
    for i, quad in enumerate(quadrilaterals):
        closed_quad = quad + [quad[0]]  # Dörtgeni kapatmak için ilk noktayı sona ekle
        xs, ys = zip(*closed_quad)
        ax.fill(xs, ys, facecolor='skyblue', alpha=0.5, edgecolor='blue', linewidth=2)
        # Alanın merkezine numara ve kategori yaz
        cx = sum(p[0] for p in quad) / 4
        cy = sum(p[1] for p in quad) / 4
        
        if categories and i < len(categories):
            ax.text(cx, cy, f"{i+1}\n{categories[i]}", fontsize=10, color='black', 
                   fontweight='bold', ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round'))
        else:
            ax.text(cx, cy, f"{i+1}", fontsize=12, color='black', fontweight='bold', ha='center', va='center')

    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title("Map View in Raw Pixel Coordinates", fontsize=14)
    ax.invert_yaxis()  # Görüntü koordinatları ile uyumlu olması için y ekseni ters çevrilir

    # Harita görseli maps klasörüne kaydedilir
    map_path = os.path.join(maps_folder, "map.jpg")
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    print(f"Saved map to: {map_path}")
    plt.show()


def create_heatmap(frame_width, frame_height, quadrilaterals, person_positions, 
                  heatmap_resolution, title, filename, show_heatmap=False, categories=None):
    """
    Isı haritası ya da basit nokta haritası oluşturur
    """
    print(f"Creating {'heatmap' if show_heatmap else 'map'}...")
    maps_folder = ensure_maps_folder_exists()
    
    # Harita için figür oluşturulur
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('white')
    
    min_x, max_x = 0, frame_width
    min_y, max_y = 0, frame_height
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    
    if show_heatmap:
        # Isı haritası için histogram ve yumuşatma işlemleri
        heatmap_size = heatmap_resolution
        x_bins = np.linspace(min_x, max_x, heatmap_size)
        y_bins = np.linspace(min_y, max_y, heatmap_size)
        # Kişi konumlarından 2D histogram oluşturulur
        heatmap, _, _ = np.histogram2d(
            [p[0] for p in person_positions],
            [p[1] for p in person_positions],
            bins=[x_bins, y_bins]
        )
        # Isı haritası daha düzgün görünsün diye gaussian filtre uygulanır
        try:
            heatmap = gaussian_filter(heatmap.astype(np.float32), sigma=2)
        except Exception as e:
            print(f"Smoothing error: {e}")
        # Renk haritası tanımlanır
        colors = [(0, 0, 1, 0), (0, 0, 1, 0.3), (0, 1, 1, 0.6), (1, 1, 0, 0.8), (1, 0, 0, 1)]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        # Isı haritası görselleştirilir
        heat = ax.imshow(
            heatmap.T,
            origin='lower',
            extent=[min_x, max_x, min_y, max_y],
            alpha=0.7,
            cmap=cmap,
            aspect='auto'
        )
        # Seçili alanlar sadece çizgi olarak gösterilir
        for i, quad in enumerate(quadrilaterals):
            closed_quad = quad + [quad[0]]
            xs, ys = zip(*closed_quad)
            ax.plot(xs, ys, 'blue', linewidth=2)
            # Alan numarası ve kategorisi ekle
            cx = sum(x for x, _ in quad) / 4
            cy = sum(y for _, y in quad) / 4
            
            if categories and i < len(categories):
                ax.text(cx, cy, f"{i+1}\n{categories[i]}", fontsize=10, color='black', 
                       fontweight='bold', ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round'))
            else:
                ax.text(cx, cy, f"{i+1}", fontsize=14, color='black', fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        # Renk çubuğu eklenir
        plt.colorbar(heat, label="People density")
    else:
        # Alanları doldur ve insan konumlarını nokta olarak çiz
        for i, quad in enumerate(quadrilaterals):
            closed_quad = quad + [quad[0]]
            xs, ys = zip(*closed_quad)
            ax.fill(xs, ys, facecolor='skyblue', alpha=0.5, edgecolor='blue', linewidth=2)
            
            # Alan numarası ve kategorisi ekle
            cx = sum(x for x, _ in quad) / 4
            cy = sum(y for _, y in quad) / 4
            
            if categories and i < len(categories):
                ax.text(cx, cy, f"{i+1}\n{categories[i]}", fontsize=10, color='black', 
                       fontweight='bold', ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round'))
            else:
                ax.text(cx, cy, f"{i+1}", fontsize=14, color='black', fontweight='bold')
                
        # İnsan konumları kırmızı noktalarla gösterilir
        xs = [p[0] for p in person_positions]
        ys = [p[1] for p in person_positions]
        ax.scatter(xs, ys, c='red', s=20, alpha=0.5, edgecolors='none')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)
    ax.invert_yaxis()  # Görüntü koordinatları ile uyumlu olması için y ekseni ters çevrilir
    # Harita görseli maps klasörüne kaydedilir
    map_path = os.path.join(maps_folder, filename)
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    print(f"Saved map to: {map_path}")
    plt.show()


def create_detections_on_image(frame, person_positions):
    """
    Orijinal görüntü üzerinde tespit edilen insan konumlarını gösterir
    """
    maps_folder = ensure_maps_folder_exists()
    # Orijinal frame üzerinde kişi konumları kırmızı noktalarla gösterilir
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.scatter(
        [p[0] for p in person_positions], 
        [p[1] for p in person_positions],
        c='red', 
        alpha=0.5,
        s=10
    )
    plt.title("People Detections on Original Image")
    # Görsel maps klasörüne kaydedilir
    map_path = os.path.join(maps_folder, "map_with_detections.jpg")
    plt.savefig(map_path, dpi=300)
    print(f"Saved map to: {map_path}")
    plt.show()


def create_category_analysis(person_positions, person_categories):
    """
    Kategori bazlı analiz grafikleri oluşturur
    """
    print("Creating category analysis...")
    maps_folder = ensure_maps_folder_exists()
    
    # Kategori sayılarını hesapla
    category_counts = Counter(person_categories)
    
    # Pasta grafiği oluştur
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Pasta grafiği
    wedges, texts, autotexts = ax1.pie(
        category_counts.values(), 
        labels=category_counts.keys(),
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        explode=[0.05] * len(category_counts),
        textprops={'fontsize': 10}
    )
    
    # Pasta grafiği düzenle
    plt.setp(autotexts, size=9, weight="bold")
    ax1.set_title('Tespitlerin Kategorilere Göre Dağılımı', fontsize=14)
    ax1.axis('equal')  # Pasta grafiğinin daire şeklinde olmasını sağlar
    
    # Çubuk grafiği
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    
    # Kategorileri tespit sayısına göre sırala
    sorted_indices = np.argsort(counts)[::-1]  # Büyükten küçüğe sırala
    sorted_categories = [categories[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    
    # Çubuk renkleri
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_categories)))
    
    # Çubuk grafiği çiz
    bars = ax2.bar(
        sorted_categories, 
        sorted_counts, 
        color=colors,
        edgecolor='black', 
        linewidth=1
    )
    
    # Çubukların üzerine değeri yaz
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f'{int(height)}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 piksel yukarıda
            textcoords="offset points",
            ha='center', 
            va='bottom', 
            fontsize=10,
            fontweight='bold'
        )
    
    ax2.set_title('Kategorilere Göre Tespit Sayıları', fontsize=14)
    ax2.set_xlabel('Kategori', fontsize=12)
    ax2.set_ylabel('Tespit Sayısı', fontsize=12)
    ax2.set_ylim(0, max(sorted_counts) * 1.1)
    
    # Kategoriler uzunsa, x ekseni etiketlerini döndür
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    plt.tight_layout()
    
    # Grafikleri kaydet
    fig_path = os.path.join(maps_folder, "category_analysis.jpg")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved category analysis to: {fig_path}")
    plt.show()
    
    # Zamana dayalı analiz (detaylı çizgisel grafik)
    time_series_analysis(person_positions, person_categories, maps_folder)


def time_series_analysis(person_positions, person_categories, maps_folder):
    """
    Kategorilere göre zamansal analiz grafiği oluşturur
    Not: Bu fonksiyon, gerçek zamanlı veri olmadığı için tespit sırasını zaman olarak kabul eder
    """
    print("Creating time series analysis...")
    
    # Kategori listesini oluştur (sırası önemli)
    categories = sorted(set(person_categories))
    
    # Her kategorinin zamansal verilerini oluştur
    time_data = {cat: [] for cat in categories}
    cumulative_counts = {cat: 0 for cat in categories}
    
    # Tüm tespitlerin %10'unu bir zaman dilimi kabul et
    num_time_points = 20  # 20 zaman noktası
    chunk_size = max(1, len(person_categories) // num_time_points)
    
    x_points = []
    
    for t in range(0, len(person_categories), chunk_size):
        time_point = t // chunk_size
        x_points.append(time_point)
        
        # Bu zaman dilimindeki kategorileri say
        chunk_categories = person_categories[t:t+chunk_size]
        chunk_counts = Counter(chunk_categories)
        
        # Her kategori için kümülatif sayıları güncelle
        for cat in categories:
            cumulative_counts[cat] += chunk_counts.get(cat, 0)
            time_data[cat].append(cumulative_counts[cat])
    
    # Çizgisel grafik oluştur
    plt.figure(figsize=(14, 8))
    
    # Her kategori için çizgi çiz
    for cat in categories:
        if max(time_data[cat]) > 0:  # Sadece veri olan kategorileri çiz
            plt.plot(x_points, time_data[cat], marker='o', linewidth=2, label=cat)
    
    plt.title('Kategorilere Göre Kümülatif Tespit Sayıları', fontsize=14)
    plt.xlabel('Zaman (Göreceli Birim)', fontsize=12)
    plt.ylabel('Kümülatif Tespit Sayısı', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    
    # Grafiği kaydet
    time_series_path = os.path.join(maps_folder, "time_series_analysis.jpg")
    plt.savefig(time_series_path, dpi=300, bbox_inches='tight')
    print(f"Saved time series analysis to: {time_series_path}")
    plt.show()