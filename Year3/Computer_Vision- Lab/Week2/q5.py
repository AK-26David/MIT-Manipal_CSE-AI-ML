import cv2
import numpy as np
def bright_spot(pix, low_limit):
    if(pix>=low_limit):
        return 255
    else:
        return 0

img = cv2.imread('deadpool-and-wolverine.jpg',0)
low_limit=200
bright_spot_vec = np.vectorize(bright_spot)
bright_spot_img = bright_spot_vec(img, low_limit)
bright_spot_img = np.clip(bright_spot_img, 0, 255).astype(np.uint8)

cv2.imshow('Brightest spot!',bright_spot_img)

cv2.waitKey(0)
cv2.destroyAllWindows()