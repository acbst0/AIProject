import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

def ensure_maps_folder_exists():
    """
    maps klasörünün var olduğundan emin olur, yoksa oluşturur
    """
    maps_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")
    if not os.path.exists(maps_folder):
        os.makedirs(maps_folder)
        print(f"Created maps folder: {maps_folder}")
    return maps_folder

def create_basic_map(frame_width, frame_height, quadrilaterals):
    """
    Temel kuşbakışı haritayı oluşturur
    """
    print("Creating bird's eye view map...")
    maps_folder = ensure_maps_folder_exists()

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

    # maps klasörüne kaydet
    map_path = os.path.join(maps_folder, "map.jpg")
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    print(f"Saved map to: {map_path}")
    plt.show()

def create_heatmap(frame_width, frame_height, quadrilaterals, person_positions, 
                  heatmap_resolution, title, filename, show_heatmap=False):
    """
    Isı haritası ya da basit nokta haritası oluşturur
    """
    print(f"Creating {'heatmap' if show_heatmap else 'map'}...")
    maps_folder = ensure_maps_folder_exists()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('white')
    
    min_x, max_x = 0, frame_width
    min_y, max_y = 0, frame_height
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Gerçek bir ısı haritası göster (seçeneğe bağlı)
    if show_heatmap:
        # Isı haritası hesapla
        heatmap_size = heatmap_resolution
        x_bins = np.linspace(min_x, max_x, heatmap_size)
        y_bins = np.linspace(min_y, max_y, heatmap_size)
        
        # Histogram oluştur
        heatmap, _, _ = np.histogram2d(
            [p[0] for p in person_positions],
            [p[1] for p in person_positions],
            bins=[x_bins, y_bins]
        )
        
        # Yumuşatma uygula
        try:
            heatmap = gaussian_filter(heatmap.astype(np.float32), sigma=2)
        except Exception as e:
            print(f"Smoothing error: {e}")
        
        # Isı haritası renklerini oluştur
        colors = [(0, 0, 1, 0), (0, 0, 1, 0.3), (0, 1, 1, 0.6), (1, 1, 0, 0.8), (1, 0, 0, 1)]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        
        # Isı haritasını göster
        heat = ax.imshow(
            heatmap.T,
            origin='lower',
            extent=[min_x, max_x, min_y, max_y],
            alpha=0.7,
            cmap=cmap,
            aspect='auto'
        )
        
        # Seçilen alanları sadece çizgi olarak göster
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
        plt.colorbar(heat, label="People density")
    else:
        # Seçilen alanları çiz
        for i, quad in enumerate(quadrilaterals):
            closed_quad = quad + [quad[0]]
            xs, ys = zip(*closed_quad)
            
            # Alanı doldur ve kenarlarını çiz
            ax.fill(xs, ys, facecolor='skyblue', alpha=0.5, edgecolor='blue', linewidth=2)
            
            # Alan numarası ekle
            cx = sum(x for x, _ in quad) / 4
            cy = sum(y for _, y in quad) / 4
            ax.text(cx, cy, f"{i+1}", fontsize=14, color='black', fontweight='bold')
        
        # İnsan konumlarını nokta olarak çiz
        xs = [p[0] for p in person_positions]
        ys = [p[1] for p in person_positions]
        ax.scatter(xs, ys, c='red', s=20, alpha=0.5, edgecolors='none')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)
    
    ax.invert_yaxis()
    
    # maps klasörüne kaydet
    map_path = os.path.join(maps_folder, filename)
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    print(f"Saved map to: {map_path}")
    plt.show()

def create_detections_on_image(frame, person_positions):
    """
    Orijinal görüntü üzerinde tespit edilen insan konumlarını gösterir
    """
    maps_folder = ensure_maps_folder_exists()
    
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
    
    # maps klasörüne kaydet
    map_path = os.path.join(maps_folder, "map_with_detections.jpg")
    plt.savefig(map_path, dpi=300)
    print(f"Saved map to: {map_path}")
    plt.show()