import cv2
import numpy as np
img = cv2.imread('deadpool-and-wolverine.jpg')
x, y = 100, 100
pixel = img[y, x]
b, g, r = pixel
print(f"RGB values at ({x}, {y}): ({r}, {g}, {b})")