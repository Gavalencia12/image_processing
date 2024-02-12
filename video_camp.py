import cv2
import numpy as np
cap = cv2.VIdeoCaptura(0)
if not (cap.isOpened()):
    print("Error al leer la camara!!!!")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out =cv2.VideoWriter('salida.avi',cv2.VIdeoWriter_fource('M','J','P','G'),10(frame_width,frame_height))