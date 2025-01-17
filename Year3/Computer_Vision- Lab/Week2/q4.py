import cv2
import numpy as np


img = cv2.imread('deadpool-and-wolverine.jpg',0)

minPixel = 90
maxPixel = 120

mask = (img >= minPixel) & (img <= maxPixel)
res = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

res[mask] = img[mask]

cv2.imshow('Gray level', res)
cv2.waitKey(0)