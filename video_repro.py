import cv2
import numpy as np

cap = cv2.VideoCapture('img/race_car.mp4')
if not (cap.isOpened()):
    print("Error al abrir el archivo de video")
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame= cv2.resize(frame,(0,0),fx=0.25, fy=0.25)
        cv2.imshow('video de carros', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

