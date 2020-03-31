import cv2

import tensorflow as tf
import numpy as np
import time
import numpy as np
import seaborn as sns
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import os
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam





CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]
#CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]




def prepare(filepath):
    IMG_SIZE = 48
   # img_array = cv2.imread(filepath)  # read in the image, convert to grayscale
    new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) 





model = tf.keras.models.load_model("C:/Users/vikra/Documents/TKS/full.model")



cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    #frame1 = frame
    frame1 = cv2.resize(frame, (200, 200))
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #IMG_SIZE = 48
    #img_array = cv2.imread(frame)  # read in the image, convert to grayscale
    #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    #new_array = frame1.reshape(-1, 48, 48, 1) 

    #frame = cv2.resize(frame, -1, 48, 48, 1)
    #cv2.imshow('Input', frame)
    #cv2.imshow('Input', frame1)

    prediction = model.predict([prepare(gray)])

    final = (CATEGORIES[int(np.argmax(prediction[0]))])
    #cv2.imshow('Input', frame1)
    
    #prediction = model.predict([new_array])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,final,(200,100), font, 1, (0,0,0), 2, cv2.LINE_AA)

    cv2.imshow('Input', frame)
    
    #print(prediction)
    #print(CATEGORIES[int(np.argmax(prediction[0]))])

    #time.sleep(1)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
