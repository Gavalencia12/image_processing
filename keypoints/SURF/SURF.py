import cv2
import numpy as np

# Función para obtener los puntos de interés y descriptores utilizando Harris y BRIEF
def detect_and_compute(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar esquinas utilizando el detector de esquinas Harris
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Umbralizar los valores de esquina para obtener los puntos de interés
    corners = cv2.dilate(corners, None)
    keypoints = np.argwhere(corners > 0.01 * corners.max())

    # Convertir los puntos de interés a flotantes y crear los objetos KeyPoint
    keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 3) for x in keypoints]

    # Calcular los descriptores utilizando el algoritmo BRIEF
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    _, descriptors = brief.compute(gray, keypoints)

    return keypoints, descriptors

# Función para realizar el matching de descriptores utilizando fuerza bruta
def match_descriptors(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# Cargar la imagen de referencia
img_reference = cv2.imread('img/object.jpg')
img_reference = cv2.resize(img_reference, (0, 0), fx=0.5, fy=0.5)
# Obtener los puntos de interés y descriptores de la imagen de referencia
keypoints_reference, descriptors_reference = detect_and_compute(img_reference)

# Iniciar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar un fotograma de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener los puntos de interés y descriptores del fotograma actual
    keypoints_frame, descriptors_frame = detect_and_compute(frame)

    # Realizar el matching de descriptores entre la imagen de referencia y el fotograma actual
    matches = match_descriptors(descriptors_reference, descriptors_frame)

    # Dibujar los matches en el fotograma actual
    img_matches = cv2.drawMatches(img_reference, keypoints_reference, frame, keypoints_frame, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Mostrar el fotograma con los matches
    cv2.imshow('Object Detection', img_matches)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()