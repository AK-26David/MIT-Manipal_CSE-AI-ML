import cv2
import numpy as np

# Start the video capture from the default camera (0)
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

    # Convert the grayscale image to float32 for cornerHarris
    gray = np.float32(gray)

    # Apply Harris corner detection
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # Dilate the corner detection result for better visualization
    dst = cv2.dilate(dst, None)

    # Threshold the corners
    frame[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark the corners in red

    # Display the frame with detected corners
    cv2.imshow('Harris Corners', frame)

    # Break the loop on 'ESC' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
