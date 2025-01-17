import cv2
import numpy as np
import glob

# Define the dimensions of the chessboard
chessboard_size = (9, 6)  # Number of inner corners per chessboard row and column
square_size = 0.025  # Size of a square in meters (adjust based on your chessboard)

# Prepare object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Load chessboard images
images = glob.glob('*.jpg')  # Adjust path

for img_path in images:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard Corners', img)

        # Wait for a key press or set a long duration
        cv2.waitKey(0)  # Wait indefinitely for a key press

cv2.destroyAllWindows()

# Step 5: Calibrate the camera
if imgpoints:  # Ensure we have found corners
    h, w = gray.shape  # Use the last processed gray image size
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    # Print camera matrix and distortion coefficients
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)

    # Save calibration results if needed
    np.savez('calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
else:
    print("No corners found in the images.")
