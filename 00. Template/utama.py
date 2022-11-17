import cv2
from handDetection import hand_tracking_module
 
handDetection = hand_tracking_module(min_detection_confidence=0.5)

webcam = cv2.VideoCapture()
webcam.open(0,cv2.CAP_DSHOW)


while True:
    status , frame = webcam.read()
    frame = cv2.flip(frame,1)
    handLandMarks = handDetection.findHandLandMarks(image=frame,draw=True)

    cv2.imshow("hand Landmark",frame)
    if cv2.waitKey(1) == ord('x'):
        break
cv2.destroyAllWindows()
webcam.release()
