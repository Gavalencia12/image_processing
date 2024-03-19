import cv2
import numpy as np

# Cargar la imagen
img = cv2.imread('img/objectq.jpg', 0)  # Leemos en escala de grises

# Crear el detector y descriptor BRIEF
star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# Detectar puntos clave en la imagen
kp = star.detect(img, None)

# Calcular los descriptores BRIEF para los puntos clave
kp, des = brief.compute(img, kp)

# Dibujar los puntos clave con sus descriptores BRIEF
img_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Mostrar la imagen con los puntos clave y descriptores
cv2.imshow('Keypoints + BRIEF Descriptors', img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()