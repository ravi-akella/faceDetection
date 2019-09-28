import numpy as np
import cv2
import time
prev_time = 0
def run_face_detect_on_frame(frame):
    global prev_time,cur_time
    #img = cv2.imread("faces.jpeg",1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    path = "haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier(path)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5,minSize=(40,40))
    print('No of faces detected is: %s'%(len(faces)))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.namedWindow("Image",cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Image",frame)
    cur_time = time.time_ns()
    time_diff = (cur_time - prev_time)*1e-6
    fps = 1e3/time_diff
    print('fps = %s'%fps)
    prev_time = cur_time
    return cv2.waitKey(1)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        _,frame = cap.read()
        key = run_face_detect_on_frame(frame)
        if key == ord('q'):
            break
        
        
    
