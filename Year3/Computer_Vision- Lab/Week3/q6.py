import cv2

# Load an image
image = cv2.imread('input.jpg')

# Apply a box filter (mean filter)
# ksize defines the size of the kernel (e.g., 5x5)
ksize = (5, 5)
smooth_box_image = cv2.blur(image, ksize)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Smooth Box Filtered Image', smooth_box_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('smooth_box_filtered_image.jpg', smooth_box_image)

