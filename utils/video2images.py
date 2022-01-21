import argparse
import cv2
import os


def parse_args(args):
    parser = argparse.ArgumentParser(description='convert model')
    parser.add_argument(
        '--video',
        help='path to video',
        type=str,
        required=True
    )
    parser.add_argument(
        '--annotation',
        help='path to annotation\'s txt file',
        type=str,
        required=True
    )
    parser.add_argument(
        '--output',
        help='path to output dir',
        type=str,
        required=True
    )
    return parser.parse_args(args)


def nake_dataset(video_path: str, annotation_path: str, save_dir: str):
    with open(annotation_path) as f:
        lines = f.readlines()
    timesteps = (list(map(int, line.replace('\n', '').split())) for line in lines)

    os.makedirs(save_dir, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()

    cur_count = 0
    count_frames_per_person = -1

    while success:
        success, image = vidcap.read()
        if cur_count > count_frames_per_person:
            cur_count = 0
            start, end, person = next(iter(timesteps))
            print('Generating frames for person %i' % person)
            os.makedirs(f'{save_dir}/{person}', exist_ok=True)
            count_frames_per_person = 120*(end - start)

        if (cur_count+1) % 120 == 0:
            cv2.imwrite(f'{save_dir}/{person}/{cur_count}.png', image)
            print('Read a new frame: ', cur_count)

        cur_count += 1


def main(args=None):
    args=parse_args(args)

    nake_dataset(args.video, args.annotation, args.output)


if __name__ == '__main__':
    main()