import cv2
import os


with open('data/gait_train.txt') as f:
  lines = f.readlines()
timesteps = (list(map(int, line.replace('\n', '').split())) for line in lines)

os.makedirs('data/frames', exist_ok=True)

vidcap = cv2.VideoCapture('data/gait_train.mp4')
success, image = vidcap.read()

cur_count = 0
count_frames_per_person = -1

while success:
    success, image = vidcap.read()
    if cur_count > count_frames_per_person:
        cur_count = 0
        start, end, person = next(iter(timesteps))
        print('Generating frames for person %i' % person)
        os.makedirs('data/frames/%i' % person, exist_ok=True)
        count_frames_per_person = 120*(end - start)

    if (cur_count+1) % 120 == 0:
        cv2.imwrite("data/frames/%i/frame%i.jpg" % (person, cur_count), image)     # save frame as JPEG file
        print('Read a new frame: ', cur_count)

    cur_count += 1