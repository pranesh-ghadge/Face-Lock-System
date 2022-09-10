import shutil
import numpy as np
import cv2
import time
import os

timeout = 20
timeout_start = time.time()
cap = cv2.VideoCapture(0)

training_image_folder_path = "D:/Data Science Projects/Project 7/model/dataset/full_images/person/"

if os.path.exists(training_image_folder_path):
    shutil.rmtree(training_image_folder_path)
    
os.mkdir(training_image_folder_path)
    

count_of_training_image = 0
while time.time() < timeout_start + timeout :
    ret, frame = cap.read()

    image = np.zeros(frame.shape, np.uint8) 
    image = frame

    cv2.imshow('frame',image)

    if cv2.waitKey(1) == ord('q'):
        break

    imagepath = training_image_folder_path + "training_image" + str(count_of_training_image) + ".png"

    cv2.imwrite(imagepath, image)
    count_of_training_image +=1