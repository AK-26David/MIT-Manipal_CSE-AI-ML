import numpy as np
import cv2
import streamlit as st
from PIL import Image

def compute_gradients(image):
    """Compute gradients using Sobel operators."""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    orientation = np.arctan2(sobel_y, sobel_x) * (180 / np.pi) % 180
    return magnitude, orientation

def compute_histogram(magnitude, orientation, cell_size=8):
    """Compute histogram of oriented gradients for each cell."""
    num_cells_y, num_cells_x = magnitude.shape[0] // cell_size, magnitude.shape[1] // cell_size
    histogram = np.zeros((num_cells_y, num_cells_x, 9))  # 9 bins for 0-180 degrees

    for y in range(num_cells_y):
        for x in range(num_cells_x):
            cell_mag = magnitude[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size]
            cell_ori = orientation[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size]
            for bin in range(9):
                lower_bound = bin * 20
                upper_bound = (bin + 1) * 20
                # Bin edges for orientation
                mask = (cell_ori >= lower_bound) & (cell_ori < upper_bound)
                histogram[y, x, bin] = np.sum(cell_mag[mask])
    return histogram

def normalize_block(histogram, block_size=2):
    """Normalize the histogram for each block."""
    num_blocks_y, num_blocks_x = histogram.shape[0] - block_size + 1, histogram.shape[1] - block_size + 1
    normalized_blocks = np.zeros((num_blocks_y, num_blocks_x, block_size*block_size*9))

    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            block = histogram[y:y+block_size, x:x+block_size].flatten()
            norm = np.sqrt(np.sum(block**2) + 1e-6)
            normalized_blocks[y, x] = block / norm
    return normalized_blocks

def extract_hog_features(image, cell_size=8, block_size=2):
    """Extract HOG features from the image."""
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    magnitude, orientation = compute_gradients(image_gray)
    histogram = compute_histogram(magnitude, orientation, cell_size)
    normalized_features = normalize_block(histogram, block_size)
    return normalized_features

def visualize_hog(hog_features, cell_size=8, block_size=2):
    """Visualize the HOG features."""
    num_blocks_y, num_blocks_x = hog_features.shape[0], hog_features.shape[1]
    hog_image = np.zeros((num_blocks_y * cell_size * block_size, num_blocks_x * cell_size * block_size), dtype=np.float32)

    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            block = hog_features[y, x].reshape(block_size, block_size, 9)
            for i in range(block_size):
                for j in range(block_size):
                    orientation_histogram = block[i, j]
                    max_orientation = np.argmax(orientation_histogram)
                    angle = max_orientation * 20
                    magnitude = np.sum(orientation_histogram)
                    y_center = y * cell_size * block_size + i * cell_size
                    x_center = x * cell_size * block_size + j * cell_size
                    length = int(magnitude * 10)  # Adjust scaling factor for visibility
                    # Draw lines for each orientation bin
                    if length > 0:
                        angle_rad = np.deg2rad(angle)
                        end_x = int(x_center + length * np.cos(angle_rad))
                        end_y = int(y_center - length * np.sin(angle_rad))
                        cv2.line(hog_image, (x_center, y_center), (end_x, end_y), (255), 1)

    # Normalize to 0-255
    hog_image = np.uint8(255 * (hog_image - np.min(hog_image)) / (np.max(hog_image) - np.min(hog_image)))
    return hog_image

def main():
    st.title("HOG (Histogram of Oriented Gradients) Feature Extraction")

    # File uploader to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Display the original image
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Extract HOG features
        hog_features = extract_hog_features(image_np)
        hog_image = visualize_hog(hog_features)

        # Convert the HOG image to a format suitable for Streamlit display
        hog_image_pil = Image.fromarray(hog_image)

        # Display the HOG image
        st.image(hog_image_pil, caption='HOG Feature Visualization', use_column_width=True)

if __name__ == "__main__":
    main()
