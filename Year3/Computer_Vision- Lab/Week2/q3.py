import cv2
import numpy as np

img=cv2.imread('deadpool-and-wolverine.jpg')
cv2.imshow('image',img)
gamma=0.5
gamma_corr=np.array(255*(img/255)**gamma, dtype='uint8')

cv2.imshow('Gamma corrected!',gamma_corr)

cv2.waitKey(0)
cv2.destroyAllWindows()