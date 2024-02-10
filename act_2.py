import cv2
import numpy as np
img = cv2.imread('img/butterfly.jpg')
num_rows, num_cols = img.shape[:2]
rattion_matrix = cv2.getRotationMatrix2D((num_cols/2,num_rows/2),30,1)
img_rotation = cv2.warpAffine(img, rotation_matrix,(num_cols,num_rows))