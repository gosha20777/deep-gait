import cv2
import os
import argparse


def parse():
    parser = argparse.ArgumentParser(description='Splitting video script')
    parser.add_argument("timesteps", type=str, help='Path to gait_train.txt')
    parser.add_argument("input_dir", type=str, help='Path to input video file (gait_train.mp4)')
    parser.add_argument("output_dir", type=str, help='Path to save output files')
    parser.add_argument("--step", type=float,  default=1, help='Length of a frame (sec)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    with open(args.timesteps) as f:
      timesteps = f.readlines()
    timesteps = (list(map(int, line.replace('\n', '').split())) for line in timesteps)

    frames = 'frames'
    os.makedirs(f'{args.output_dir}/{frames}', exist_ok=True)

    vidcap = cv2.VideoCapture(args.input_dir)
    success, image = vidcap.read()

    cur_count = 0
    count_frames_per_person = -1
    while success:
        success, image = vidcap.read()
        if cur_count > count_frames_per_person:
            cur_count = 0
            start, end, person = next(iter(timesteps))
            print('Generating frames for person %i' % person)
            os.makedirs(f'{args.output_dir}/{frames}/{person}', exist_ok=True)
            count_frames_per_person = 120*(end-start)

        if (cur_count+1) % round((120 * args.step)) == 0:
            cv2.imwrite(f'{args.output_dir}/{frames}/{person}/{cur_count}.png', image)     # save frame as JPEG file
            print('Read a new frame: ', cur_count)
        cur_count += 1
