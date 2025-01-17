import cv2
img = cv2.imread('deadpool-and-wolverine.jpg')
new_width = 640
new_height = 480
resized_img = cv2.resize(img, (new_width, new_height))
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()