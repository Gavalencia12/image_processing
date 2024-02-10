import cv2
import numpy as np

#Rotation of the image
img = cv2.imread('img/butterfly.jpg')
num_rows, num_cols = img.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((num_cols/2,num_rows/2),30,1)
img_rotation = cv2.warpAffine(img, rotation_matrix,(num_cols,num_rows))

cv2.imshow('Rotation',img_rotation)
cv2.imwrite("img/img_rotation.jpg", img_rotation)




#rotation of the video

cap = cv2.VideoCapture(0)

#counter for the new windows
num = int(input("How many Windows do you want?: "))
ventanas = []


for i in range(num):
    ventana_= f'Video {i}'
    ventanas.append(ventana_)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
      # video = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      video = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      video = cv2.warpAffine(frame, rotation_matrix,(num_cols,num_rows))
      # video = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
      # video = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # video = cv2.resize(video, (0, 0), fx=0.5, fy=0.5)
      for ventana in ventanas:
            cv2.imshow(ventana, video)
      '''
        Close the video with the following keys
        Number ASII       Key
        113               q
        81                Q
        120               x
        88                X
        27                ESC
        13                Enter
      '''
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q') or key == ord('Q') or key == ord('x') or key == ord('X') or key == 27 or key == 13:
        break
    else:
      break

cap.release()
cv2.destroyAllWindows()