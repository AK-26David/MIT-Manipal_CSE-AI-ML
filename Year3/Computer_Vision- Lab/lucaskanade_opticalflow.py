import numpy as np
import cv2

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Take the first frame from the camera and convert it to grayscale
ret, old_frame = cap.read()
if not ret:
    print("Error: Could not read the frame from webcam.")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect good features to track (using Shi-Tomasi corner detection)
feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7)  # Increased maxCorners
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create random colors for visualizing the optical flow tracks
color = np.random.randint(0, 255, (100, 3))

# Create a mask image for drawing the motion tracks
mask = np.zeros_like(old_frame)

while True:
    # Capture the next frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the new frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the optical flow using the Lucas-Kanade method
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points (status 1 means the flow was found)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # Convert the coordinates to integers
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        
        # Draw the line showing movement between points
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        
        # Draw a circle on the current position of the point
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

    # Overlay the optical flow tracks on the frame
    output = cv2.add(frame, mask)

    # Display the resulting frame
    cv2.imshow('Lucas-Kanade Optical Flow', output)

    # Update the previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Break on ESC key press
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
