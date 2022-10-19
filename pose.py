import mediapipe as mp
import cv2

#import drawing utilities
mp_drawing = mp.solutions.drawing_utils

#import holistic mediapipe model
mp_holistic = mp.solutions.holistic

#Gets webcam feed

cap = cv2.VideoCapture(0)

#initiate holistic model
with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():

        #Writes feed to screen
        ret, frame = cap.read()

        #Recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Make detections
        results = holistic.process(image)
        
        #face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        #recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

        #right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        #left hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        #pose detection
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        #lables frame
        cv2.imshow('Raw Webcame Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

