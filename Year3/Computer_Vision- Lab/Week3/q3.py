import cv2
import numpy as np

def max_filter(image, kernel_size=3):
    return cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8))

# Load an image in grayscale
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Max filter
max_filtered_image = max_filter(image, kernel_size=3)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Max Filtered Image', max_filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('max_filtered_image.jpg', max_filtered_image)

