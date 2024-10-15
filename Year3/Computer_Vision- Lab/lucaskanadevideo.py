import cv2
import numpy as np
import os

# Specify the directory path
directory = "/Users/arnavkarnik/Documents/MIT-Manipal_CSE-AI-ML/Year3/Computer_Vision- Lab"

# List all files in the directory
files = os.listdir(directory)

# Filter for video files (you may need to add more extensions if necessary)
video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov'))]

if not video_files:
    print(f"No video files found in {directory}")
    exit()

print("Available video files:")
for i, file in enumerate(video_files):
    print(f"{i + 1}. {file}")

# Ask user to select a file
selection = int(input("Enter the number of the file you want to use: ")) - 1
video_path = os.path.join(directory, video_files[selection])

# Load the video capture
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Couldn't open video file: {video_path}")
    print(f"VideoCapture.isOpened() returned: {cap.isOpened()}")
    exit()

# Read the first frame
ret, frame = cap.read()

# Check if the frame was read successfully
if not ret:
    print("Error: Couldn't read the first frame. Exiting...")
    cap.release()
    exit()

print("Video opened successfully!")
print(f"Frame shape: {frame.shape}")

# Convert the frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Define the parameters for the Lucas-Kanade method
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Define the feature parameters
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Define the feature points to track
p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(frame)

while True:
    # Read the next frame
    ret, frame = cap.read()
    
    # Check if the frame was read successfully
    if not ret:
        print("End of video reached. Exiting...")
        break
    
    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow using the Lucas-Kanade method
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
    
    # Overlay the mask on the frame
    img = cv2.add(frame, mask)
    
    # Display the output
    cv2.imshow('Tracking', img)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Update the previous frame and points
    gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

print("Video processing completed.")