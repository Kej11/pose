#import libraries
import mediapipe as mp
import cv2
import csv
import os
import numpy as np

#calculated total number of coordinates
#num_coords = len(results.pose_landmarks.landmark) + len(results.face_landmarks.landmark)
num_coords = 501

landmarks = ['class']
for val in range(1,num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

landmarks
#writes landmarks to the CSV
with open('coords.csv', mode='w', newline='') as f:
                csv_writer = csv.writer(f,delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)

class_name = "Happy"

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
        image.flags.writeable = False

        #Make detections
        results = holistic.process(image)
        
        #face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        #recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )

        #right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        #left hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        #pose detection
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

        #export coordinates
        try:
            #extract pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            #extract facelandmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            row = pose_row+face_row
            row.insert(0, class_name)
            

            with open('coords.csv', mode='a') as t:
                csv.writer = csv.writer(f,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
                csv.writer.writerow(row)
        
        except:
            pass

        #lables frame
        cv2.imshow('Raw Webcame Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()



