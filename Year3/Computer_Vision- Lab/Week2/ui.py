import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

def load_image():
    global img, img_path, img_gray
    img_path = filedialog.askopenfilename()
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image(img)

def show_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    panel.config(image=img_tk)
    panel.image = img_tk

def apply_transformation():
    if img_path:
        transformation = transformation_var.get()
        if transformation == "Negative Image":
            result = 255 - img
        elif transformation == "Log Transformed Image":
            c = 255 / (np.log(1 + np.max(img_gray)))
            log_transformed = c * np.log(1 + img_gray)
            result = cv2.merge([log_transformed, log_transformed, log_transformed]).astype(np.uint8)
        elif transformation == "Gamma Corrected Image (gamma=2.2)":
            gamma = 2.2
            gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
            result = gamma_corrected
        elif transformation == "Gray Level Slicing":
            min_val, max_val = 90, 120
            gray_sliced = np.where((img_gray >= min_val) & (img_gray <= max_val), 255, 0).astype(np.uint8)
            result = cv2.merge([gray_sliced, gray_sliced, gray_sliced])
        elif transformation == "Brightest Spot":
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img_gray)
            result = cv2.circle(img.copy(), max_loc, 10, (0, 0, 255), 2)
        elif transformation == "Histogram Graph":
            plt.hist(img_gray.ravel(), 256, [0, 256])
            plt.title('Histogram of the Image')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.show()
            return
        show_image(result)

app = tk.Tk()
app.title("Image Transformation Tool")

transformation_var = tk.StringVar()
transformation_var.set("Select Transformation")

dropdown_menu = ttk.Combobox(app, textvariable=transformation_var)
dropdown_menu['values'] = ["Negative Image", "Log Transformed Image", "Gamma Corrected Image (gamma=2.2)",
                           "Gray Level Slicing", "Brightest Spot", "Histogram Graph"]
dropdown_menu.pack(pady=20)

load_button = tk.Button(app, text="Load Image", command=load_image)
load_button.pack(pady=10)

apply_button = tk.Button(app, text="Apply Transformation", command=apply_transformation)
apply_button.pack(pady=10)

panel = tk.Label(app)
panel.pack(pady=20)

app.mainloop()