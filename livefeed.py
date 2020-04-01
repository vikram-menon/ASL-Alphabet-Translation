import cv2
import tensorflow as tf
import numpy as np





CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]
#CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]




def prepare(filepath):
    IMG_SIZE = 48
    new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) 





model = tf.keras.models.load_model("file_path_to_downloaded_model")



cap = cv2.VideoCapture(1)#use 0 if using inbuilt webcam

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame1 = cv2.resize(frame, (200, 200))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    prediction = model.predict([prepare(gray)])
    final = (CATEGORIES[int(np.argmax(prediction[0]))])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,final,(200,100), font, 1, (0,0,0), 2, cv2.LINE_AA)

    cv2.imshow('Input', frame)
    

    c = cv2.waitKey(1)
    if c == 27: # hit esc key to stop
        break

cap.release()
cv2.destroyAllWindows()
