# This file will be minimum opencv file for mirroring the webcam

print("This is mirror.py")

import cv2 as cv
import numpy as np

# Load Haar cascade (vi facedetect not working)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

first = True
score = 0

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Mirror image
    frame = cv.flip(frame, 1)
    height, width = frame.shape[:2]

    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    
    if first:
        prev_gray = gray.copy()
        first = False

    
    delta = cv.absdiff(gray, prev_gray).sum()

    # Draw rectangle
    cv.rectangle(frame, (0, 0), (100, 100), (100, 100, 100), 2)

    
    cv.putText(frame, str(delta), (50, 300),
               cv.FONT_HERSHEY_SIMPLEX,
               2,  # font scale
               (255, 255, 255),  # color
               4)  # thickness

    # Face detection using Haar cascade 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
   

    # Display score
    cv.putText(frame, f"Score: {score}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    
    cv.imshow('Mirror Color Camera', frame)

    # Quit on 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    
    prev_gray = gray.copy()

# Release resources
cap.release()
cv.destroyAllWindows()# mirror.py
print("This is mirror.py")

import cv2 as cv
import numpy as np

# Load Haar cascade 
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

first = True
score = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Mirror image
    frame = cv.flip(frame, 1)
    height, width = frame.shape[:2]

    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Store first frame for delta calculation
    if first:
        prev_gray = gray.copy()
        first = False

    # Calculate change in gray
    delta = cv.absdiff(gray, prev_gray).sum()

    # Draw static rectangle
    cv.rectangle(frame, (0, 0), (100, 100), (100, 100, 100), 2)

    # Display delta value
    cv.putText(frame, str(delta), (50, 300),
               cv.FONT_HERSHEY_SIMPLEX,
               2,  # font scale
               (255, 255, 255),  # color
               4)  # thickness

    # Face detection using Haar cascade 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    

    # Display score
    cv.putText(frame, f"Score: {score}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    
    cv.imshow('Mirror Color Camera', frame)

    # Quit on 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = gray.copy()

# Release resources
cap.release()
cv.destroyAllWindows()
