import cv2
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Kullanıcıya video seçtiren ve seçilen videonun yolunu döndüren fonksiyon
# Arayüzde video listesi, önizleme ve seçim butonları bulunur
# Seçim yapılmazsa program sonlanır
def select_video():
    videos_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
    
    # videos klasöründeki video dosyalarını bul
    video_files = [f for f in os.listdir(videos_folder) if f.endswith((".mp4", ".avi", ".mov"))]
    
    if not video_files:
        print("No videos found in the videos folder!")
        exit()
    
    # Tkinter arayüzünü başlat
    root = tk.Tk()
    root.title("Video Selection")
    root.geometry("800x600")
    root.configure(bg="#f0f0f0")
    
    root.eval('tk::PlaceWindow . center')
    
    style = ttk.Style()
    style.theme_use('clam')
    
    # Başlık çubuğu
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
    
    # İçerik çerçevesi (liste ve önizleme)
    content_frame = tk.Frame(root, bg="#f0f0f0", padx=20, pady=20)
    content_frame.pack(fill=tk.BOTH, expand=True)
    
    selected_video = tk.StringVar()
    
    # Video önizleme alanı
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
    
    # Video listesi alanı
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
        # Seçilen videonun ilk karesinden önizleme oluşturur
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
    
    # Video dosyalarını listeye ekle (isim, boyut, süre)
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
        # Listeden video seçilince önizleme güncellenir
        selected_item = tree.selection()
        if selected_item:
            video_name = tree.item(selected_item[0])['values'][0]
            video_path = os.path.join(videos_folder, video_name)
            create_thumbnail(video_path)
    
    tree.bind("<<TreeviewSelect>>", on_tree_select)
    
    def on_select():
        # Seç butonuna basınca seçilen video döndürülür
        selected_item = tree.selection()
        if selected_item:
            video_name = tree.item(selected_item[0])['values'][0]
            selected_video.set(os.path.join(videos_folder, video_name))
            root.destroy()
    
    # Butonlar (Vazgeç ve Seç)
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
    
    # Varsayılan olarak ilk video seçili ve önizlemesi gösterilir
    if video_files:
        tree.selection_set(tree.get_children()[0])
        video_path = os.path.join(videos_folder, video_files[0])
        create_thumbnail(video_path)
    
    root.mainloop()
    
    if not selected_video.get():
        print("No video selected!")
        exit()
    
    return selected_video.get()