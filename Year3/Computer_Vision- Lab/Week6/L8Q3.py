import cv2
import numpy as np

def klt_tracking(video_path):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    # Convert to grayscale
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Detect good features to track
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

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

        # Select good points
        if p1 is not None and st is not None:
            good_new = p1[st.flatten() == 1]
            good_old = p0[st.flatten() == 1]

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

            img = cv2.add(frame, mask)
        else:
            img = frame  # Just show the current frame if no points found

        # Display the result
        cv2.imshow('KLT Feature Tracking', img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        # Update previous frame and points
        old_gray = new_gray.copy()
        p0 = good_new.reshape(-1, 1, 2) if good_new.size > 0 else p0

    cap.release()
    cv2.destroyAllWindows()

# Replace 'path_to_video.mp4' with your video file path
klt_tracking('cycling.mp4')
