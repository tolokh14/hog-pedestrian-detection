from imutils.object_detection import non_max_suppression
import numpy as np
import cv2 as cv
import datetime as dt
import urllib.request as req

URL = 'http://192.168.100.108/cam-hi.jpg'

#initialize the HOG descriptor/person detector
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cv.startWindowThread()

#open webcaam video stream
# cap = cv.VideoCapture(2)
cap = cv.VideoCapture(URL + ":81/stream")

#start the timer
fps_start_time = dt.datetime.now()
fps = 0
total_frames = 0

#the output will be written to output .avi
out = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc(*'MJPG'),15.,(640,480))

while(cap.isOpened()):
    #reading the frame
    ret, frame = cap.read()

    #resizing for faster detection
    frame = cv.resize(frame, (640,480))

    total_frames += 1

    #using grayscale picture, also for faster detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # blur = cv.GaussianBlur(gray, (5,5), 0)
    # _, thresh = cv.threshold(blur, 150, 255, cv.THRESH_BINARY)
    
    #detect people in the image
    #returns the bounding boxes for the detected objects
    #winStride=(4, 4),padding=(8, 8), scale=1.05
    boxes, weights = hog.detectMultiScale(gray,winStride=(8,8))

    boxes = np.array([[x, y, x + w, y + h] for (x,y,w,h) in boxes])
    pick = non_max_suppression(boxes, probs=None, overlapThresh=0.65)

    c = 1
    for (xA, yA, xB, yB) in pick:
        #display the detected boxes in the colour picture
        cv.rectangle(frame, (xA, yA), (xB,yB), (0,255,0), 2)
        cv.putText(frame, 'Person', (xA + 5, yA - 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        c += 1
    #write the output video
    out.write(frame.astype('uint8'))

    fps_end_time = dt.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)
    
    fps_text = "FPS: {:.2f}".format(fps)

    cv.putText(frame, fps_text, (5, 30), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
    cv.putText(frame, f'Total Persons : {c - 1}', (20, 450), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

    #displaying the frame
    cv.imshow("Frame", frame)
    # cv.imshow("vid", thresh)
    if cv.waitKey(1) & 0xFF == ord('q'):
        #breaking the loop if the user types q
        #note that the video windows must be higlighted
        break
#when everything done, release the capture
cap.release()
#and release the output
out.release()
#finally
cv.destroyAllWindows()
cv.waitKey(1)