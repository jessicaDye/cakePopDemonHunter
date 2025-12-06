# This file will be minimum opencv file for mirroring the webcam


# mirror.py
print("This is mirror.py")

import cv2 as cv
import numpy as np
import time
import random

# Load Haar cascade (vifacedetect not working)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

first = True
score = 0

# Corner demon variables
demon_size = 60
corner_demon_active = False
corner_demon_timer = 0
corner_demon_duration = 2  
corner_demon_x, corner_demon_y = 0, 0

# Falling demon variables
fall_size = 50
initial_fall_speed = 5   # made slower for first demon
fall_speed = initial_fall_speed
fall_y = -fall_size       # starts above screen for slow fall
fall_x = 0                
game_over = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv.flip(frame, 1)
    height, width = frame.shape[:2]

    # Initialize falling demon x-position once screen width is known
    if fall_x == 0:
        fall_x = width // 2

    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if first:
        prev_gray = gray.copy()
        first = False

    delta = cv.absdiff(gray, prev_gray).sum()

    cv.rectangle(frame, (0, 0), (100, 100), (100, 100, 100), 2)
    cv.putText(frame, str(delta), (50, 300),
               cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

    # Face Detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_boxes = []
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_boxes.append((x, y, w, h))

    # Corner demon
    if not corner_demon_active and random.random() < 0.01:  
        corner_demon_active = True
        corner_demon_timer = time.time()
        corner_demon_x = random.choice([0, width - demon_size])
        corner_demon_y = random.choice([0, height - demon_size])

    if corner_demon_active:
        cv.rectangle(frame,
                     (corner_demon_x, corner_demon_y),
                     (corner_demon_x + demon_size, corner_demon_y + demon_size),
                     (0, 0, 255), -1)  # filled corner demon

        demon_now = gray[corner_demon_y:corner_demon_y + demon_size,
                         corner_demon_x:corner_demon_x + demon_size]
        demon_prev = prev_gray[corner_demon_y:corner_demon_y + demon_size,
                               corner_demon_x:corner_demon_x + demon_size]
        demon_motion = cv.absdiff(demon_now, demon_prev).sum()

        if demon_motion > 500000:
            score += 1
            corner_demon_active = False  # disappear after hit

        if time.time() - corner_demon_timer > corner_demon_duration:
            corner_demon_active = False

    # Falling demon
    if not game_over:
        # Demons are blue for increased visibility
        cv.rectangle(frame,
                     (fall_x, fall_y),
                     (fall_x + fall_size, fall_y + fall_size),
                     (255, 0, 0), -1)
        fall_y += fall_speed

        # Check collision with face
        for (fx, fy, fw, fh) in face_boxes:
            if (fall_x < fx + fw and
                fall_x + fall_size > fx and
                fall_y < fy + fh and
                fall_y + fall_size > fy):
                game_over = True

        # If demon reaches bottom safely
        if fall_y > height:
            score += 5
            fall_y = -fall_size          
            fall_x = np.random.randint(50, width - 50)
            fall_speed = 12              

    # Game Over
    if game_over:
        cv.putText(frame, "GAME OVER",
                   (width // 2 - 150, height // 2),
                   cv.FONT_HERSHEY_SIMPLEX, 2,
                   (0, 0, 255), 4)

    # Score
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
