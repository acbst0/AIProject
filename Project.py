import cv2
import matplotlib.pyplot as plt
import numpy as np

# Video dosyasını aç ve ilk kareyi al
cap = cv2.VideoCapture('camera.mp4')
ret, frame = cap.read()
cap.release()

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
    x_unit = x_vec / np.linalg.norm(x_vec)
    y_unit = y_vec / np.linalg.norm(y_vec)
    z_unit = z_vec / np.linalg.norm(z_vec)
    B = np.vstack([x_unit, y_unit, z_unit]).T
    return B, origin_vec

def resolve_to_3d(p, B, origin_vec):
    vec = np.array(p) - origin_vec
    coords_3d, _, _, _ = np.linalg.lstsq(B, vec, rcond=None)
    return coords_3d

B, origin_vec = compute_3d_basis(origin, x_axis, y_axis, z_axis)

projected_quads = []
for quad in quadrilaterals:
    projected_quad = []
    for p in quad:
        x, y, z = resolve_to_3d(p, B, origin_vec)
        projected_quad.append((x, y))
    projected_quads.append(projected_quad)


def angle_between_points(org, p1, p2):
    """
    Verilen üç nokta arasında, orijinden iki nokta arasındaki açıyı döndürür.
    origin: Orijin noktası (0, 0)
    p1: Birinci nokta (x1, y1)
    p2: İkinci nokta (x2, y2)
    """
    # Eğer negatifse pozitife çevir
    if org[0] < 0:
        org = (abs(org[0]), org[1])
    if org[1] < 0:
        org = (org[0], abs(org[1]))
    
    v1 = np.array(p1) - np.array(org)
    v2 = np.array(p2) - np.array(org)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # [-1, 1] aralığında kısıtlama
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# --- DÜZ DÖRTGENE DÖNÜŞTÜRME ---
def make_rectangle_from_quad(quad):
    angle = angle_between_points(origin, x_axis, y_axis)
    print(f"Açı: {angle:.2f} derece")
    p0, p1, p2, p3 = quad
    p0 = np.array(p0)
    v1 = np.array(p1) - p0
    v2 = np.array(p3) - p0
    width = np.linalg.norm(v1)
    height = np.linalg.norm(v2)
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    r0 = p0
    r1 = r0 + u1 * width
    r2 = r1 + u2 * height
    r3 = r0 + u2 * height
    # z = 0 olarak 3D hale getiriyoruz
    # noktaları açı değişkeni ile göre saat tersine döndür
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    r0 = np.dot(rotation_matrix, r0) + origin_vec
    r1 = np.dot(rotation_matrix, r1) + origin_vec
    r2 = np.dot(rotation_matrix, r2) + origin_vec
    r3 = np.dot(rotation_matrix, r3) + origin_vec

    
    return [(r0[0], r0[1], 0), (r1[0], r1[1], 0), (r2[0], r2[1], 0), (r3[0], r3[1], 0)]


rectified_quads = []
for quad in projected_quads:
    rect_quad = make_rectangle_from_quad(quad)
    rectified_quads.append(rect_quad)

# --- DÜZENLENMİŞ HARİTAYI ÇİZ ---
fig, ax = plt.subplots()
for i, quad in enumerate(rectified_quads, start=1):
    quad_np = np.array(quad + [quad[0]])
    ax.plot(quad_np[:, 0], quad_np[:, 1], 'b-')
    cx = np.mean(quad_np[:, 0])
    cy = np.mean(quad_np[:, 1])
    ax.text(cx, cy, f"{i}", fontsize=10, color='red')

ax.set_aspect('equal')
ax.set_title("Kuşbakışı Harita (Dikdörtgene Düzenlenmiş)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.savefig("map_rectified.jpg", dpi=300)
plt.show()

# --- 3D'den tekrar 2D'ye projeksiyon ---
def project_3d_to_2d(p3d, B, origin_vec):
    return B @ p3d + origin_vec

fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
for i, quad in enumerate(rectified_quads, start=1):
    projected_2d = [project_3d_to_2d(p, B, origin_vec) for p in quad]
    xs, ys = zip(*projected_2d + [projected_2d[0]])
    ax.plot(xs, ys, 'cyan')
    cx = np.mean(xs)
    cy = np.mean(ys)
    ax.text(cx, cy, f"#{i}", color='cyan', fontsize=12)
ax.set_title("Dikdörtgene Düzenlenmiş Görüntü Üzerinde")
plt.show()
