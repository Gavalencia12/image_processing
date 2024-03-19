import cv2
import numpy as np
import matplotlib.pyplot as plt

#sift
sift = cv2.SIFT_create()

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck= True)

img = cv2.imread('img/facial.jpg')
img1 = cv2.imread('img/facial.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)



keypoints, descriptors = sift.detectAndCompute(img,None)
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

matches = bf.match(descriptors,descriptors_1)
matches = sorted(matches, key = lambda x:x.distance)

img2 = cv2.drawMatches(img, keypoints, img1, keypoints_1,matches[300:600], img1, flags=2)

cv2.imshow('SIFT', img2)
cv2.waitKey(0)