import cv2
import matplotlib.pyplot as plt
import numpy as np

def select_quadrilaterals(frame):
    """
    Frame üzerinde dörtgen alanları seçmemizi sağlayan fonksiyon
    """
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
    
    return quadrilaterals