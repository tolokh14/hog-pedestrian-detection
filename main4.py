import numpy as np
from imutils.object_detection import non_max_suppression
import cv2 as cv
import datetime as dt
import requests

'''
INFO SECTION
- if you want to monitor raw parameters of ESP32CAM, open the browser and go to http://192.168.x.x/status
- command can be sent through an HTTP get composed in the following way http://192.168.x.x/control?var=VARIABLE_NAME&val=VALUE (check varname and value in status)
'''

# ESP32 URL
URL = 'http://192.168.100.108'
AWB = True

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

fps_start_time = dt.datetime.now()
fps = 0
total_frames = 0

# Face recognition and opencv setup
cap = cv.VideoCapture(URL + ":81/stream")

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

if __name__ == '__main__':
    set_resolution(URL, index=7)

    while True:
        if cap.isOpened():
            ret, frame = cap.read()
            total_frames += 1
            if ret:
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

            elif key == ord('q'):
                val = int(input("Set quality (10 - 63): "))
                set_quality(URL, value=val)

            elif key == ord('a'):
                AWB = set_awb(URL, AWB)

            elif key == 27:
                break

    cv.destroyAllWindows()
    cap.release()