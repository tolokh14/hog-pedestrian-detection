import numpy as np
from imutils.object_detection import non_max_suppression
import cv2 as cv
import requests

def cameraCapture(cam):
    cap = cv.VideoCapture(cam)
    while(cap.isOpened()):
        #reading the frame
        ret, frame = cap.read()

        #using grayscale picture, also for faster detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        
        boxes, weights = hog.detectMultiScale(gray,winStride=(8,8))

        boxes = np.array([[x, y, x + w, y + h] for (x,y,w,h) in boxes])
        pick = non_max_suppression(boxes, probs=None, overlapThresh=0.65)

        c = 1
        for (xA, yA, xB, yB) in pick:
            #display the detected boxes in the colour picture
            cv.rectangle(frame, (xA, yA), (xB,yB), (0,255,0), 2)
            cv.putText(frame, 'Person', (xA + 5, yA - 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            c += 1

        cv.putText(frame, f'Total Persons : {c - 1}', (20, 450), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

        #displaying the frame
        cv.imshow("Frame", frame)
        # cv.imshow("vid", thresh)
        if cv.waitKey(1) & 0xFF == ord('q'):
            #breaking the loop if the user types q
            #note that the video windows must be higlighted
            break

URL = 'http://192.168.100.106'

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cv.startWindowThread()

# Face recognition and opencv setup
cam1 = cameraCapture(0)
cam2 = cameraCapture(URL+':81/stream')


#when everything done, release the capture
cam1.release()
cam2.release()
#finally
cv.destroyAllWindows()
cv.waitKey(1)