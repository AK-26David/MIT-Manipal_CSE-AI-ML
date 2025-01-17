import cv2
import numpy as np

# Function to initialize the tracking
def initialize_tracking(video_path):
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    # Convert to grayscale
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Define a region of interest (ROI) for tracking
    roi = cv2.selectROI("Select Object to Track", first_frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = roi
    p0 = np.array([[x + w // 2, y + h // 2]], dtype=np.float32)  # Initial point for tracking

    # Create a mask for drawing
    mask = np.zeros_like(first_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Lucas-Kanade method
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None)

        # Check if points were found
        if p1 is not None and st is not None:
            # Select good points
            good_new = p1[st.flatten() == 1]
            good_old = p0[st.flatten() == 1]

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)  # Convert to integer
                c, d = old.ravel().astype(int)   # Convert to integer
                mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

            img = cv2.add(frame, mask)
        else:
            img = frame  # Just show the current frame if no points found

        # Display the result
        cv2.imshow('Object Tracking', img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        # Update the previous frame and previous points
        old_gray = new_gray.copy()
        p0 = good_new.reshape(-1, 1, 2) if good_new.size > 0 else p0  # Only update if good_new is not empty

    cap.release()
    cv2.destroyAllWindows()

# Replace 'path_to_video.mp4' with your video file path
initialize_tracking('cycling.mp4')
