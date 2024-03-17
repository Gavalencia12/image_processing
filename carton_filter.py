import cv2
import numpy as np

def cartoonize_image(img, ds_factor=4, sketch_mode=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 7)
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)
    num_repetitions = 10
    sigma_color = 5
    sigma_space = 7
    size = 5

    for i in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, size, sigma_color, sigma_space)
        img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR)
        dst = np.zeros(img_gray.shape)
        dst = cv2.bitwise_and(img_output, img_output, mask=mask)
        return dst
    
cap = cv2.VideoCapture(0)
if not (cap.isOpened()):
    print("¡ERROR AL LEER LA CÁMARA!")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
#out = cv2.VideoWriter('img/videos/video.avi', cv2.VideoWriter_fourcc('M','J','P','G'),la velocidad de reproducion del video 20,(frame_width, frame_height))
out = cv2.VideoWriter('img/videos/video.avi', cv2.VideoWriter_fourcc('M','J','P','G'),20,(frame_width, frame_height))

while(True):
    ret, frame = cap.read()
    if ret == True:
        out.write(frame)
        frame_out = cartoonize_image(frame, ds_factor=2, sketch_mode=False) #True o False
        out.write(frame_out)
        cv2.imshow('Video carton', frame_out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == ord('x') or key == ord('X') or key == 27 or key == 13:
            break
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()