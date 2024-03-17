import cv2
import numpy as np

img1 = np.zeros((400, 600), dtype=np.uint8)
img1[100:300, 200:400] = 255
cv2.imshow('image Artificial', img1)

# cv2.imwrite('img/cuadro.jpg', img1)
img2 = cv2.circle(img1, (300, 200), 125, 255, -1)
cv2.imshow('image_2', img2)

img3 = cv2.bitwise_and(img1, img2)

cv2.waitKey(0)
cv2.destroyAllWindows()