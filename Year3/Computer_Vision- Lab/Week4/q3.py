import cv2

# Load an image in grayscale
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
# The two thresholds are used for the hysteresis procedure
lower_threshold = 50
upper_threshold = 150
edges = cv2.Canny(image, lower_threshold, upper_threshold)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Canny Edges', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('canny_edges.jpg', edges)

