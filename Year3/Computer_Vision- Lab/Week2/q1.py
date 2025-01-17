import cv2

img=cv2.imread('deadpool-and-wolverine.jpg')
cv2.imshow('image',img)
neg_image=255-img

cv2.imshow('Negative Image!',neg_image)

cv2.waitKey(0)
cv2.destroyAllWindows()