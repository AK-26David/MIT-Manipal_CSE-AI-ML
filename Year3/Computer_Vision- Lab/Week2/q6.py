import cv2
from matplotlib import pyplot as plt

# reads an input image
img = cv2.imread('deadpool-and-wolverine.jpg',0)

# Plot the histogram
plt.hist(img.flatten(), 256, [0, 256])

plt.title('Histogram of the Image')

plt.xlabel('Intensity')

plt.ylabel('Frequency')

plt.show()