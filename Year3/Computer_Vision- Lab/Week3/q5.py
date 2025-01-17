import cv2
import numpy as np

def sobel_sharpen(image):
    # Convert the image to grayscale if it's not already
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the Sobel operator to find the gradients in the x and y direction
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x direction
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y direction

    # Calculate the magnitude of the gradient
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))  # Convert to 8-bit image

    # Sharpen the image by adding the original image to the Sobel magnitude
    sharpened_image = cv2.addWeighted(gray_image, 1.5, sobel_magnitude, -0.5, 0)

    return sharpened_image

# Load an image
image = cv2.imread('input.jpg')

# Apply Sobel sharpening
sharpened_image = sobel_sharpen(image)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', sharpened_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('sharpened_image.jpg', sharpened_image)

