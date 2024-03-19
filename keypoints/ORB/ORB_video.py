import cv2

# Crear el detector y descriptor ORB
orb = cv2.ORB_create()

# Iniciar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer un fotograma de la cámara
    ret, frame = cap.read()

    # Detectar puntos clave y calcular descriptores ORB en el fotograma
    kp, des = orb.detectAndCompute(frame, None)

    # Dibujar los puntos clave ORB en el fotograma
    img_kp = cv2.drawKeypoints(frame, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Mostrar el fotograma con los puntos clave ORB
    cv2.imshow('Camera', img_kp)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()