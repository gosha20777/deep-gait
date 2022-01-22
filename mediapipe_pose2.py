import matplotlib.pyplot as plt
import numpy as np
import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1,
                    min_detection_confidence=0.6,  
                    min_tracking_confidence=0.6)

cap = cv2.VideoCapture('data/gait_train.mp4')
# cap = cv2.VideoCapture(0)
pTime = 0

count_frame = 0 

x = []
y = []
while True:
    success, img = cap.read()
    # img = cv2.resize(img, (256, 256))

    img_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rbg)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # mp_drawing.plot_landmarks(
        #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            # print(lm.x, lm.y, lm.z)
            # print(id, lm)
            x.append(lm.x)
            y.append(lm.y)
            

    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"fps: {str(int(fps))}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, f"frame: {str(count_frame)}", (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    count_frame +=1 
    if count_frame == 500:
        break
    
    cv2.imshow("Mediapipe", img)

    cv2.waitKey(1)



plt.plot(x, y)
plt.show()