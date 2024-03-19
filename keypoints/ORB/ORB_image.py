import cv2
import numpy as np

# Cargar la imagen
img = cv2.imread('img/object.jpg')

# Crear el detector y descriptor ORB
orb = cv2.ORB_create()

# Detectar puntos clave y calcular descriptores ORB en la imagen
kp, des = orb.detectAndCompute(img, None)

# Dibujar los puntos clave ORB en la imagen
img_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Mostrar la imagen con los puntos clave ORB
cv2.imshow('ORB Keypoints', img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()