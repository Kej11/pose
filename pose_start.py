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