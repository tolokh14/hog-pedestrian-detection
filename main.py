import cv2 as cv
import imutils

#initializing the HOG human detector
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

# load the video
cap = cv.VideoCapture('test.mp4')

# looping for the video
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        frame = imutils.resize(frame,width=min(600, frame.shape[1]))

        #detect all the region that has pedestrians
        (regions, _) = hog.detectMultiScale(frame,winStride=(4,4),padding=(4,4),scale=1.05)

        #draw the regions in the video
        for (x, y, w, h) in regions:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        cv.imshow('Video',frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

cv.destroyAllWindows()