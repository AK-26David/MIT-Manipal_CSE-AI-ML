import cv2
import numpy as np

def compute_gradients(gray):
    # Compute gradients using Sobel operator
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return sobel_x, sobel_y

def compute_harris_response(sobel_x, sobel_y, k=0.04, block_size=2):
    # Compute products of derivatives
    Ix2 = sobel_x**2
    Iy2 = sobel_y**2
    Ixy = sobel_x * sobel_y

    # Initialize Harris response matrix
    height, width = sobel_x.shape
    R = np.zeros((height, width))

    # Apply Gaussian filter to smooth the products
    gaussian_kernel = cv2.getGaussianKernel(ksize=block_size*2+1, sigma=1.0)
    gaussian_kernel = gaussian_kernel @ gaussian_kernel.T
    Ix2 = cv2.filter2D(Ix2, -1, gaussian_kernel)
    Iy2 = cv2.filter2D(Iy2, -1, gaussian_kernel)
    Ixy = cv2.filter2D(Ixy, -1, gaussian_kernel)

    # Compute Harris corner response
    for i in range(height):
        for j in range(width):
            M = np.array([[Ix2[i, j], Ixy[i, j]], [Ixy[i, j], Iy2[i, j]]])
            R[i, j] = np.linalg.det(M) - k * (np.trace(M)**2)
    
    return R

# Start video capture from default camera (0)
cap = cv2.VideoCapture(0)

# Set the video resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to float64 for precision
    gray = np.float64(gray)

    # Compute gradients
    sobel_x, sobel_y = compute_gradients(gray)

    # Compute Harris response
    R = compute_harris_response(sobel_x, sobel_y)

    # Threshold the corners
    corners = (R > 0.01 * R.max())
    frame[corners] = [0, 0, 255]  # Mark the corners in red

    # Display the frame with detected corners
    cv2.imshow('Harris Corners', frame)

    # Break the loop on 'ESC' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
