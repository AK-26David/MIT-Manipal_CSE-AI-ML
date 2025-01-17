import cv2
import numpy as np

img=cv2.imread('deadpool-and-wolverine.jpg',0)

img_float = np.float32(img)

# log transform
c = 255/(np.log(1+np.max(img_float)))
log_transformed = c*np.log(1+img_float)

# specify datatype
log_transformed=np.array(log_transformed, dtype=np.uint8)

cv2.imshow('Log transformed!',log_transformed)

cv2.waitKey(0)
cv2.destroyAllWindows()