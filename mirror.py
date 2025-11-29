# This file will be minimum opencv file for mirroring the webcam

print("This is mirror.py")

import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

first = True

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert to grayscale for first frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if first:
        prev_gray = gray.copy()
        first = False

    # Mirror image
    frame = cv.flip(frame, 1)
    height, width = frame.shape[:2]

    # Convert to grayscale again after flip
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calc change in gray
    delta = cv.absdiff(gray, prev_gray).sum()

    # Make rectangle
    cv.rectangle(frame, (0, 0), (100, 100), (100, 100, 100), 2)

    # Display text
    cv.putText(frame, str(delta), (50, 300),
               cv.FONT_HERSHEY_SIMPLEX,
               2,  # font scale
               (255, 255, 255),  # COLOR
               4)  # THICKNESS

    # Display frame
    cv.imshow('frame', frame)

    # Quit on 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # Save previous frame
    prev_gray = gray

# When done release the capture
cap.release()
cv.destroyAllWindows()

