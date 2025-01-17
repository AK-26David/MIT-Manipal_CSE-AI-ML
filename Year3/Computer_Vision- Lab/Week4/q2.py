import cv2

# Load an image in grayscale
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# Apply global thresholding
# The second argument is the threshold value
# The third argument is the maximum value to use with the THRESH_BINARY thresholding
threshold_value = 127
_, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image', binary_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('binary_image.jpg', binary_image)

