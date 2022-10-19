import mediapipe as mp
import cv2

#import drawing utilities
mp_drawing = mp.solutions.drawing_utils

#import holistic mediapipe model
mp_holistic = mp.solutions.holistic

#Gets webcam feed
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
cap = cv2.VideoCapture(0)
while cap.isOpened():

    #Writes feed to screen
    ret, frame = cap.read()
    #lables frame
    cv2.imshow('Raw Webcame Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

