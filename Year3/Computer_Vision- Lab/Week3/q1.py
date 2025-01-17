import cv2
import numpy as np
from matplotlib import pyplot as plt

def calculate_cdf(histogram):
    cdf = histogram.cumsum()
    cdf_normalized = cdf / cdf[-1]  # Normalize to the range [0, 1]
    return cdf_normalized

def histogram_specification(source_img, reference_img):
    # Convert the images to grayscale
    source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

    # Calculate histograms and cumulative distribution functions (CDFs)
    source_hist, _ = np.histogram(source_gray.flatten(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference_gray.flatten(), 256, [0, 256])

    source_cdf = calculate_cdf(source_hist)
    reference_cdf = calculate_cdf(reference_hist)

    # Create a lookup table to map pixel values from the source image to the reference image
    lookup_table = np.zeros(256)
    for i in range(256):
        closest_value = np.argmin(np.abs(reference_cdf - source_cdf[i]))
        lookup_table[i] = closest_value

    # Apply the lookup table to the source image
    specified_img = cv2.LUT(source_gray, lookup_table.astype(np.uint8))

    return specified_img

# Load the source and reference images
source_img = cv2.imread('source.jpg')
reference_img = cv2.imread('reference.jpg')

# Perform histogram specification
specified_img = histogram_specification(source_img, reference_img)

# Display the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
plt.title('Source Image')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB))
plt.title('Reference Image')

plt.subplot(1, 3, 3)
plt.imshow(specified_img, cmap='gray')
plt.title('Specified Image')

plt.show()

# Save the result
cv2.imwrite('specified_image.jpg', specified_img)

