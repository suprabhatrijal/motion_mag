import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture("pole_out1.mp4")

# Get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Create an empty array to store the vertical scan
vertical_scan = np.zeros((frame_height, frame_width), dtype=np.uint8)

scan = []

# Iterate over the frames of the video
while cap.isOpened():
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Read the next frame
    ret, frame = cap.read()

    # If the frame was not read successfully, break
    if not ret:
        break

    height, width = frame.shape[:2]
    # Create a vertical slice of the image
    slice = frame[height//10-1:height//10, :]
    slice = slice.reshape(width,3)
    scan.append(slice)
    # Wait for a key press
    key = cv2.waitKey(1)

    # If the key is `q`, break
    if key == ord("q"):
        break

scan = np.array(scan)
scan = np.rot90(scan)


cv2.imwrite("pulse.png", scan)


# Release the capture object
cap.release()

# Close all windows
cv2.destroyAllWindows()

