# This file will be minimum opencv file for mirroring the webcam


print("This is mirror.py")

import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 	
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv.flip(frame, 1)
    height, width = frame.shape[:2]

    # Convert to grayscale 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.putText(frame, "Hello", (50, 50),
    			cv.FONT_HERSHEY_SIMPLEX,
    			1,
    			(255, 0, 255), #COLOR
    			2) #THICKNESS

    # Display the resulting frame
    #cv.imshow('frame', gray)
    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break
  
    # Display frame
    #cv.imshow('Mirrored Color Camera', gray)

    # Quit when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When done release the capture
cap.release()
cv.destroyAllWindows()

