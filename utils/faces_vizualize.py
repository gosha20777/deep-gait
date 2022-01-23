import cv2
import argparse


def parse():
    parser = argparse.ArgumentParser(description='Imshow video script')
    parser.add_argument("timesteps", type=str, help='Path to annotation.txt')
    parser.add_argument("video_dir", type=str, help='Path to input video file')
    parser.add_argument("faces_dir", type=str, help='Path to folder containing pictures of faces')
    parser.add_argument("--step", type=float,  default=1, help='Length of a frame (sec)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    with open(args.timesteps) as f:
      timesteps = f.readlines()
    timesteps = (list(map(int, line.replace('\n', '').split())) for line in timesteps)

    vidcap = cv2.VideoCapture(args.video_dir)
    success, image = vidcap.read()

    cur_count = 0
    count_frames_per_person = -1
    while success:
        success, image = vidcap.read()

        if cur_count > count_frames_per_person:
            cur_count = 0
            start, end, person_id = next(iter(timesteps))

            person_img = cv2.imread(f'{args.faces_dir}/{person_id}.jpg')
            img_height, img_width, _ = person_img.shape

            print('Processing for person %i' % person_id)
            count_frames_per_person = 120*(end-start)

        if (cur_count+1) % round((120 * args.step)) == 0:

            frame_width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            x = int(frame_width*0.7)
            y = int(frame_height*0.1)

            image[y:y + img_height, x:x + img_width] = person_img
            cv2.imshow('frame', image)
            if cv2.waitKey(20) & 0xFF == 27:
                break

        cur_count += 1

    vidcap.release()
    cv2.destroyAllWindows()