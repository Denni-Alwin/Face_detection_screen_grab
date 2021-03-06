import cv2
from PIL import ImageGrab#For screen grab >> pip install pillow
import numpy as np

faceCascade = cv2.CascadeClassifier('C:\\Users\\Denni\\Desktop\\haarcascade_frontalface_default.xml')#xml file for face detection

while True:
    printscreen =  np.array(ImageGrab.grab(bbox=(5,5,1280,720)))#Grabs the screen using 1280x720 you can change it if needed
    gray = cv2.cvtColor(printscreen,cv2.COLOR_BGR2GRAY)#changes to gray
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(printscreen, (x, y), (x+w, y+h), (0,0,255), 2)
    cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))#show as window and changes GRAY 2 RGB
    if cv2.waitKey(1) & 0xFF == ord('q'):# (Q) to quit the window
        break
    
cv2.destroyAllWindows()
