from operator import index
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2 as cv
import datetime as dt
import urllib.request as req
import requests

URL = 'http://192.168.100.108'
AWB = True

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

def set_resolution(url: str, index: int=7, verbose: bool=False):
    try:
        if verbose:
            resolutions = "10: UXGA(1600x1200)\n9: SXGA(1280x1024)\n8: XGA(1024x768)\n7: SVGA(800x600)\n6: VGA(640x480)\n5: CIF(400x296)\n4: QVGA(320x240)\n3: HQVGA(240x176)\n0: QQVGA(160x120)"
            print("available resolutions\n{}".format(resolutions))

        if index in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
            requests.get(url + "/control?var=framesize&val={}".format(index))
        else:
            print("Wrong index")
    except:
        print("SET_RESOLUTION: something went wrong")

def set_quality(url: str, value: int=10, verbose: bool=False):
    try:
        if value >= 10 and value <=63:
            requests.get(url + "/control?var=quality&val={}".format(value))
    except:
        print("SET_QUALITY: something went wrong")

def set_awb(url: str, awb: int=1):
    try:
        awb = not awb
        requests.get(url + "/control?var=awb&val={}".format(1 if awb else 0))
    except:
        print("SET_QUALITY: something went wrong")
    return awb

#the output will be written to output .avi
# out = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc(*'MJPG'),15.,(640,480))

if __name__ == '__main__':
    set_resolution(URL, index=7)

    while(cap.isOpened()):
        #reading the frame
        ret, frame = cap.read()

        #resizing for faster detection
        # frame = cv.resize(frame, (640,480))

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
        # out.write(frame.astype('uint8'))

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
        key = cv.waitKey(1)

        if key == ord('r'):
            idx = int(input("Select resolution index: "))
            set_resolution(URL, index=idx, verbose=True)

        elif key == ord('k'):
            val = int(input("Set quality (10 - 63): "))
            set_quality(URL, value=val)

        elif key == ord('a'):
            AWB = set_awb(URL, AWB)

        if cv.waitKey(1) & 0xFF == ord('q'):
            #breaking the loop if the user types q
            #note that the video windows must be higlighted
            break
#when everything done, release the capture
cap.release()
#and release the output
# out.release()
#finally
cv.destroyAllWindows()
cv.waitKey(1)