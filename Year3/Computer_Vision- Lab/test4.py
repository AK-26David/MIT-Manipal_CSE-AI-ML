import numpy as np
import cv2

def harris_corner_detection(image, k=0.04, window_size=3, threshold=1e-5):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Ix = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyy = Iy ** 2

    corner_response = np.zeros_like(gray_image, dtype=np.float64)
    half_w = window_size // 2
    for y in range(half_w, gray_image.shape[0] - half_w):
        for x in range(half_w, gray_image.shape[1] - half_w):
            Sxx = np.sum(Ixx[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1])
            Sxy = np.sum(Ixy[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1])
            Syy = np.sum(Iyy[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1])

            M = np.array([[Sxx, Sxy],
                          [Sxy, Syy]])

            R = np.linalg.det(M) - k * (np.trace(M) ** 2)

            corner_response[y, x] = R

    corners = np.zeros_like(corner_response)
    corners[corner_response > threshold] = 255

    return corners

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply Harris corner detection
    corners = harris_corner_detection(frame)

    # Convert corners to RGB and overlay on the original frame
    corners_rgb = cv2.cvtColor(corners.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    output_frame = cv2.addWeighted(frame, 0.7, corners_rgb, 0.3, 0)

    # Display the result
    cv2.imshow('Harris Corners', output_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
