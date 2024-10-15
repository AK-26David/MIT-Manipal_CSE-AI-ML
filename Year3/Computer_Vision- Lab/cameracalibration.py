import cv2
import numpy as np

# Set up the termination criteria for cornerSubPix algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for a 9x6 chessboard (adjust if your chessboard has a different size)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load the image (replace with your image path)
image_path = 'chess.jpg'
img = cv2.imread(image_path)

# Check if the image was loaded correctly
if img is None:
    print(f"Image {image_path} could not be loaded.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

if ret:
    objpoints.append(objp)

    # Refine corner locations
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners on the image
    cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
    cv2.imshow('Chessboard Corners', img)
    cv2.waitKey(0)
else:
    print("Chessboard corners not found.")
    exit()

cv2.destroyAllWindows()

# Perform camera calibration using the object points and image points
ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the camera matrix
print("Camera Matrix:\n", camera_matrix)
