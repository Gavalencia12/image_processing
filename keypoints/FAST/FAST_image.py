import cv2
import numpy as np
# Cargar la imagen
img = cv2.imread('img/flor.png', 0)  # Leemos en escala de grises

# Crear el detector FAST
fast = cv2.FastFeatureDetector_create()

# Detectar puntos clave en la imagen
kp = fast.detect(img, None)

# Dibujar los puntos clave en la imagen
img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 0, 255))

# Mostrar la imagen con los puntos clave
cv2.imshow('FAST Keypoints', img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()