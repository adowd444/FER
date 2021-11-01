import cv2
import numpy as np
import os
import keras
from keras.models import load_model


dicts = ['empty', 'neutral', 'happy', 'sad', 'neutral', 'happy', 'sad']

# To capture video from a webcam
cap = cv2.VideoCapture(0)

if not (cap.isOpened()):
    print('Could not open video device')

# Load the cascade
face_cascade = cv2.CascadeClassifier('c:/users/devin/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')
model = keras.models.load_model('model_13')
counts = {}
while True:
    # Read the frame
    _, frame = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        fc = frame[y:y+w, x:x+w]
        # crop over resize
        fin = cv2.resize(fc, (224, 224))
        roi = cv2.resize(fc, (224, 224))
        roi = np.expand_dims(roi, axis=0)
        pred = model.predict(roi)
        rounded_prediction = np.argmax(pred, axis=1)
        emotion = dicts[rounded_prediction[0]]
        cv2.putText(frame, str(emotion), (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    if cv2.waitKey(1) == 27:
        break
    cv2.imshow('Filter', frame)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()