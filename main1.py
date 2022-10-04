import numpy as np
import cv2 as cv

#initialize the HOG descriptor/person detector
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cv.startWindowThread()

#open webcaam video stream
cap = cv.VideoCapture('test.mp4')

#the output will be written to output .avi
out = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc(*'MJPG'),15.,(640,480))

while(cap.isOpened()):
    #reading the frame
    ret, frame = cap.read()

    #resizing for faster detection
    frame = cv.resize(frame, (640,480))

    #using grayscale picture, also for faster detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresh, None, iterations=3)
    
    #detect people in the image
    #returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(blur,winStride=(8,8))

    boxes = np.array([[x, y, x + w, y + h] for (x,y,w,h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        #display the detected boxes in the colour picture
        cv.rectangle(frame, (xA, yA), (xB,yB), (0,255,0), 2)

    #write the output video
    out.write(frame.astype('uint8'))

    #displaying the frame
    cv.imshow("Frame", frame)
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