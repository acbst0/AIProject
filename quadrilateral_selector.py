import cv2
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import simpledialog

# Bu fonksiyon, kullanıcıya bir görüntü (frame) üzerinde dörtgen alan(lar) seçtiren arayüzü başlatır.
# Her 4 tıklama bir dörtgen oluşturur. ENTER'a basınca seçim tamamlanır.
def select_quadrilaterals(frame):
    """
    Frame üzerinde dörtgen alanları seçmemizi sağlayan fonksiyon
    """
    quadrilaterals = []
    categories = []
    current_quad = []

    # Matplotlib ile görsel arayüz oluşturulur
    fig2, ax2 = plt.subplots()
    ax2.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Area Selection: Every 4 clicks make a quadrilateral.\nPress ENTER when finished.")

    # Kategori girmek için Tkinter root
    root = tk.Tk()
    root.withdraw()  # Ana pencereyi gizle

    def quad_onclick(event):
        # Her tıklamada bir köşe eklenir, 4 olunca dörtgen çizilir ve kaydedilir
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
                
                # Kullanıcıdan kategori adı iste
                category = simpledialog.askstring("Kategori", 
                                                 f"Seçilen alan #{quad_id} için kategori adı giriniz:", 
                                                 parent=root)
                
                if not category:
                    category = f"Area {quad_id}"
                
                categories.append(category)
                
                cx = sum(x for x, _ in current_quad) / 4
                cy = sum(y for _, y in current_quad) / 4
                ax2.text(cx, cy, f"#{quad_id}\n{category}", color='white', fontsize=12, 
                        weight='bold', ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))
                
                quadrilaterals.append(current_quad.copy())
                current_quad.clear()
                plt.draw()

    def onkey(event):
        # ENTER'a basınca seçim tamamlanır ve pencere kapanır
        if event.key == 'enter':
            plt.close()
            root.destroy()

    fig2.canvas.mpl_connect('button_press_event', quad_onclick)
    fig2.canvas.mpl_connect('key_press_event', onkey)
    plt.show()
    
    return quadrilaterals, categories