import cv2

# Cargar el clasificador de cuerpos y caras pre-entrenado
body_cascade = cv2.CascadeClassifier('xml/haarcascade_fullbody.xml')
face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')

# Iniciar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer un fotograma de la cámara
    ret, frame = cap.read()

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar cuerpos y caras en el fotograma
    bodies = body_cascade.detectMultiScale(gray, 1.1, 3)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Dibujar rectángulos alrededor de los cuerpos y caras detectados
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Detectar y dibujar puntos clave FAST en el fotograma
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(gray, None)
    img_kp = cv2.drawKeypoints(gray, kp, None, color=(255, 0, 0))

    # Combinar el fotograma con los rectángulos y los puntos clave FAST
    combined = cv2.bitwise_or(frame, img_kp)

    # Mostrar el fotograma resultante
    cv2.imshow('Camera', combined)

    # Salir si se presiona la tecla 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == ord('x') or key == ord('X') or key == 27 or key == 13:
      break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()