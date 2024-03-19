import cv2
import imutils

# Cargar la imagen de referencia
img_reference = cv2.imread('img/object.jpg')

# Convertir la imagen de referencia a escala de grises
gray_reference = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)

# Inicializar el detector SURF
surf = cv2.xfeatures2d.SURF_create()

# Detectar puntos de interés y descriptores en la imagen de referencia
keypoints_reference, descriptors_reference = surf.detectAndCompute(gray_reference, None)

# Iniciar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar un fotograma de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el fotograma a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar puntos de interés y descriptores en el fotograma actual
    keypoints_frame, descriptors_frame = surf.detectAndCompute(gray_frame, None)

    # Realizar el matching de descriptores entre la imagen de referencia y el fotograma actual
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_reference, descriptors_frame, k=2)

    # Filtrar los matches usando el ratio de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Dibujar los matches en el fotograma actual
    img_matches = cv2.drawMatches(img_reference, keypoints_reference, frame, keypoints_frame, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Mostrar el fotograma con los matches
    cv2.imshow('Object Detection', imutils.resize(img_matches, width=800))

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
