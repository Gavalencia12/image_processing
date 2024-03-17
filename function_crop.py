import cv2
import numpy as np

img = cv2.imread("img/flor.png",cv2.IMREAD_GRAYSCALE)
cv2.imshow("origin flor", img)
print(img.shape) #show dimension
img_crop = img[50:180, 30:400]

cv2.imshow("Image",img_crop)
cv2.waitKey(0)
cv2.destroyAllWindows
