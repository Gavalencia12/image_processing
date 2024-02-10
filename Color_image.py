import cv2
print([x for x in dir(cv2) if x.startswith('COLOR_')])

img = cv2.imread('img/cube.jpg')

img2 = cv2.resize(img,(0,0),fx=0.5, fy=0.5)
img3 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

cv2.imshow('Image of cube Rubick in grey',img3)

cv2.imwrite("img/Big_cube_grey.jpg", img3)
cv2.waitKey()