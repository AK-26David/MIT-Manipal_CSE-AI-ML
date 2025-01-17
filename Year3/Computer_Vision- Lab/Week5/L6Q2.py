import cv2
import numpy as np
import streamlit as st


def compute_gradients(image):
    """Compute gradients using Sobel operators."""
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return grad_x, grad_y


def compute_harris_response(grad_x, grad_y, k=0.04):
    """Compute Harris response for corner detection."""
    Ixx = grad_x ** 2
    Ixy = grad_x * grad_y
    Iyy = grad_y ** 2

    # Apply Gaussian filter to smooth the products
    Ixx = cv2.GaussianBlur(Ixx, (3, 3), 0)
    Ixy = cv2.GaussianBlur(Ixy, (3, 3), 0)
    Iyy = cv2.GaussianBlur(Iyy, (3, 3), 0)

    # Compute Harris response
    det = (Ixx * Iyy) - (Ixy ** 2)
    trace = Ixx + Iyy
    R = det - k * (trace ** 2)

    return R


def harris_corner_detection(frame, quality_level):
    """Apply Harris corner detection on the frame with a quality level."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)

    grad_x, grad_y = compute_gradients(gray)
    R = compute_harris_response(grad_x, grad_y)

    # Normalize the Harris response
    R_normalized = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX)
    R_normalized = np.uint8(R_normalized)

    # Convert quality level to a threshold
    threshold = quality_level * 255
    corners = np.zeros_like(R_normalized)
    corners[R_normalized > threshold] = 255

    # Create a copy of the frame for output
    output_frame = frame.copy()

    # Draw corners on the frame
    for y in range(corners.shape[0]):
        for x in range(corners.shape[1]):
            if corners[y, x] == 255:
                cv2.circle(output_frame, (x, y), 3, (0, 0, 255), -1)

    return output_frame


def main():
    st.title("Harris Corner Detection on Live Video")

    # Slider for quality level
    quality_level = st.slider("Quality Level", 0.0, 1.0, 0.01)

    # Start video capture
    cap = cv2.VideoCapture(0)

    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        # Apply Harris corner detection
        processed_frame = harris_corner_detection(frame, quality_level)

        # Convert BGR to RGB for Streamlit display
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        stframe.image(processed_frame, channels="RGB", use_column_width=True)

    cap.release()


if __name__ == "__main__":
    main()
