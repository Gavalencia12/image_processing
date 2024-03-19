import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

#sift
sift = cv2.SIFT_create()

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck= True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    #read images
    suc, img = cap.read()
    img1 = img

    start = time.time()


    keypoints, descriptors = sift.detectAndCompute(img,None)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

    matches = bf.match(descriptors,descriptors_1)
    matches = sorted(matches, key = lambda x:x.distance)

    end = time.time()
    totaltime = end-start

    fps = 1 / totaltime
    #print("FPS: ",fps)
    img2 = cv2.drawMatches(img, keypoints, img1, keypoints_1, matches, None, flags=2)

    cv2.putText(img2,f'FPS: {int(fps)}',(20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),3)
    cv2.imshow('SIFT_VIDEO', img2)

    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q') or key == ord('Q') or key == ord('x') or key == ord('X') or key == 27 or key == 13:
        break
