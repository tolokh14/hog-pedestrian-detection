import numpy as np
from imutils.object_detection import non_max_suppression
import cv2 as cv

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cv.startWindowThread()

cap1 = cv.VideoCapture('http://192.168.100.119:81/stream')
cap2 = cv.VideoCapture('http://192.168.100.106:81/stream')
while 1:

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray1 = cv.equalizeHist(gray1)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    gray2 = cv.equalizeHist(gray2)

    boxes1, weights1 = hog.detectMultiScale(gray1,winStride=(8,8))
    boxes2, weights2 = hog.detectMultiScale(gray2,winStride=(8,8))

    boxes1 = np.array([[x, y, x + w, y + h] for (x,y,w,h) in boxes1])
    pick1 = non_max_suppression(boxes1, probs=None, overlapThresh=0.65)

    boxes2 = np.array([[x, y, x + w, y + h] for (x,y,w,h) in boxes2])
    pick2 = non_max_suppression(boxes2, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in pick1:
        #display the detected boxes in the colour picture
        cv.rectangle(frame1, (xA, yA), (xB,yB), (0,255,0), 2)
        cv.putText(frame1, 'Person', (xA + 5, yA - 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    for (xA, yA, xB, yB) in pick2:
        #display the detected boxes in the colour picture
        cv.rectangle(frame2, (xA, yA), (xB,yB), (0,255,0), 2)
        cv.putText(frame2, 'Person', (xA + 5, yA - 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv.imshow("Frame 1", frame1)
    cv.imshow("Frame 2", frame2)
    if cv.waitKey(1) & 0xFF == ord('q'):
        #breaking the loop if the user types q
        #note that the video windows must be higlighted
        break

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap1.release()
cap2.release()
cv.destroyAllWindows()