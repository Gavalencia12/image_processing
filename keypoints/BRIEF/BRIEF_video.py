import cv2
import numpy as np

# Crear el detector y descriptor BRIEF
star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# Iniciar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer un fotograma de la cámara
    ret, frame = cap.read()

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar puntos clave en el fotograma
    kp = star.detect(gray, None)

    # Calcular los descriptores BRIEF para los puntos clave
    kp, des = brief.compute(gray, kp)

    # Dibujar los puntos clave con sus descriptores BRIEF
    img_kp = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Mostrar el fotograma con los puntos clave y descriptores
    cv2.imshow('Camera', img_kp)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()