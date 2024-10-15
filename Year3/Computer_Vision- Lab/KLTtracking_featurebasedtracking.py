import numpy as np
import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or replace with the camera index

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame from the camera
ret, old_frame = cap.read()
if not ret:
    print("Error: Could not read the frame from camera.")
    exit()

# Convert to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect initial feature points to track
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask for drawing purposes
mask = np.zeros_like(old_frame)  # Create a mask with the same size and type as the frame

while True:
    # Capture the next frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the frame from camera.")
        break

    # Convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        
        # Draw the line
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        # Draw a circle
        frame = cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)

    # Overlay the tracking mask on the current frame
    img = cv2.add(frame, mask)

    # Display the resulting frame
    cv2.imshow('KLT Tracking', img)

    # Update the previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Break on ESC key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
