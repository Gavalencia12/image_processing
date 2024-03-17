import cv2
import numpy as np

face_cascade_path = "xml/haarcascade_frontalface_default.xml"
eye_cascade_path = "xml/haarcascade_eye.xml"
mouth_cascade_path = "xml/haarcascade_mcs_mouth.xml"
leftear_cascade_path = "xml/haarcascade_mcs_leftear.xml"
rightear_cascade_path = "xml/haarcascade_mcs_rightear.xml"
nose__cascade_path = "xml/haarcascade_mcs_nose.xml"

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
leftear_cascade = cv2.CascadeClassifier(leftear_cascade_path)
rightear_cascade = cv2.CascadeClassifier(rightear_cascade_path)
nose__cascade =cv2.CascadeClassifier(nose__cascade_path)

if face_cascade.empty():
    raise IOError('Cannot load Filter')


cap = cv2.VideoCapture(0)
ds_factor = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=ds_factor, fy= ds_factor, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    mouth = mouth_cascade.detectMultiScale(gray, 1.3, 5)
    leftear = leftear_cascade.detectMultiScale(gray,1.3, 5)
    rightear = rightear_cascade.detectMultiScale(gray,1.3,5)
    nose = nose__cascade.detectMultiScale(gray,1.3,5)
    
    for(x, y, w, h) in faces:
        cv2.putText(frame, 'Cabeza', (x+w,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[ y:y+h , x:x+w ]
        roi_color = frame[ y:y+h, x:x+w ]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            center = (int(x_eye + 0.5 * w_eye),int(y_eye + 0.5* h_eye))
            radius = int(0.3*(w_eye + h_eye))
            color = (0,255,0)
            thickness = 3
            cv2.circle(roi_color, center, radius, color, thickness)
            if center[0] < w // 2:
                cv2.putText(roi_color, 'Ojo Izquierdo', (center[0] - 50, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                cv2.putText(roi_color, 'Ojo Derecho', (center[0] - 50, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    for (x, y, w, h) in mouth:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(frame, 'Boca', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    for (x, y, w, h) in leftear:        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(frame, 'Right ear', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    
    for (x, y, w, h) in rightear:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(frame, 'Left ear', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    for (x, y, w, h) in nose:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, 'Nariz', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('Detecter of face and eyes',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == ord('x') or key == ord('X') or key == 27 or key == 13:
        break
    
cap.release()
cv2.destroyAllWindows()