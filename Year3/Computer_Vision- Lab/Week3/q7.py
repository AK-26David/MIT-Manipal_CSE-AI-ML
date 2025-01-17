import cv2

# Load an image
image = cv2.imread('input.jpg')

# Apply a Gaussian filter
# ksize defines the size of the Gaussian kernel (e.g., 5x5)
# sigmaX is the standard deviation in the x direction
ksize = (5, 5)
sigmaX = 0  # 0 means that sigma will be calculated based on ksize
smooth_gaussian_image = cv2.GaussianBlur(image, ksize, sigmaX)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Gaussian Smoothed Image', smooth_gaussian_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('smooth_gaussian_image.jpg', smooth_gaussian_image)

