import cv2
video_file_path = '/home/student/Desktop/220962021_CV/Lab1/MIND-BLOWING VISUAL EFFECTS Compilation _ Curlykidlife.mp4'
cap = cv2.VideoCapture(video_file_path)
if not cap.isOpened():
    print("Error opening video file")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video FPS: {fps}")
    print(f"Video Frame Size: {frame_width}x{frame_height}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
        cv2.imshow('Video', frame)

        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the VideoCapture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()