#Show a image with OpenCV

import cv2

#The image is stored in the variable
img = cv2.imread('img/butterfly.jpg')

#Adjust the image size
img_2 = cv2.resize(img,(0,0),fx=1.5, fy=1.5)

#Show the image
cv2. imshow('Monarch Butterfly ', img_2)

cv2.waitKey()