import cv2

cap = cv2.VideoCapture(0)

#counter for the new windows
num = int(input("How many Windows do you want?: "))
ventanas = []


for i in range(num):
    _ventana= f'Video {i}'
    ventanas.append(_ventana)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
      # video = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      video = cv2.GaussianBlur(frame, (3,3), 0)
      '''
      video gray
      '''
      video = cv2.Canny (image = frame, threshold1=100,threshold2=100)
      
      
      # video = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
      # video = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # video = cv2.resize(video, (0, 0), fx=0.5, fy=0.5)
      for ventana in ventanas:
            cv2.imshow(ventana, video)
      '''
        Close the video with the following keys
        Number ASII       Key
        113               q
        81                Q
        120               x
        88                X
        27                ESC
        13                Enter
      '''
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q') or key == ord('Q') or key == ord('x') or key == ord('X') or key == 27 or key == 13:
        break
    else:
      break

cap.release()
cv2.destroyAllWindows()