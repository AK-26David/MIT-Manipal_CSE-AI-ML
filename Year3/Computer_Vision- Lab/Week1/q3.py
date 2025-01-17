import cv2
import numpy as np
img = cv2.imread('image.jpg')
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
img = np.zeros((512, 512, 3), np.uint8)
cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), 2)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()