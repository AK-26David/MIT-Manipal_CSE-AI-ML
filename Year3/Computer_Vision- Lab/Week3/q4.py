import cv2

# Load an image in grayscale
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Median filter
median_filtered_image = cv2.medianBlur(image, ksize=3)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Median Filtered Image', median_filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('median_filtered_image.jpg', median_filtered_image)

