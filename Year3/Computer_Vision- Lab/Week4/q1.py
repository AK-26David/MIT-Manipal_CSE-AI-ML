import cv2
import numpy as np

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)

    # Calculate the difference between the original image and the blurred version
    sharpened = float(amount + 1) * image - float(amount) * blurred

    # Clip the values to be in the valid range [0, 255] and convert back to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    if threshold > 0:
        # Only enhance areas with differences above the threshold
        low_contrast_mask = np.abs(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened

# Load an image
image = cv2.imread('input.jpg')

# Apply unsharp masking
sharpened_image = unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', sharpened_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('sharpened_image.jpg', sharpened_image)

