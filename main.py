import sys
import time
from datetime import datetime

import numpy as np

from model.runnner import VisionModel
import research.lane_change as lc
import atexit
import cv2
import math

# open up our test file
cap = cv2.VideoCapture('test_video/test_creditview.mp4')

out_traffic = open('traffic_output.txt', "a+")

vision_model = VisionModel(using_wide=False, show_vis=True)

field_experiment = False

vis_frames = []

cam_frames = []

ts = np.array([[1.42070485, 0.0, -30.16740088],
                  [0.0, 1.42070485, 91.030837],
                  [0.0, 0.0, 1.0]])


def log_traffic_info(lead_x, lead_y, lead_d, veh_speed):
    date_time = datetime.now()

    info = "Date: " + str(date_time) + ", Type: Lead Vehicle" + ", Distance x: " + str(lead_x) + ", Distance y: " + \
           str(lead_y) + ", Distance_t: " + str(lead_d) + " m " + ", Vehicle Speed: " + str(veh_speed * 3.6) + "\n"

    out_traffic.write(info)


def res_frame_2(frame):
    return cv2.resize(frame, (1164, 874), interpolation=cv2.INTER_NEAREST)


def res_frame_3(frame):
    return cv2.warpPerspective(frame, ts, (1164, 874), flags=cv2.INTER_LINEAR)


def res_frame(frame):
    frame1_shape = frame.shape

    if frame1_shape[0] != 874 and frame1_shape[1] != 1168:
        if frame1_shape[0] >= 874 and frame1_shape[1] >= 1168:

            crop_vertical = (frame1_shape[0] - 874) // 2
            crop_horizontal = (frame1_shape[1] - 1164) // 2

            return frame[crop_vertical:frame1_shape[0] - crop_vertical,
                   crop_horizontal:frame1_shape[1] - crop_horizontal]
        else:
            raise Exception("This image source cannot be used for model!")

    return frame


def setup_image_stream():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


def get_frames():
    ret1, frame_1 = cap.read()
    # time.sleep(0.025)
    ret2, frame_2 = cap.read()

    if not (ret1 or ret2):
        raise Exception("Error reading from image source")

    frame_1 = res_frame_2(frame_1)
    frame_2 = res_frame_2(frame_2)

    cam_frames.append(frame_1)
    cam_frames.append(frame_2)

    return frame_1, frame_2


def process_model(frame1, frame2):
    global field_experiment

    lead_x, lead_y, lead_d, pose_speed, vis_image = vision_model.run_model(frame1, frame2)

    log_traffic_info(lead_x, lead_y, lead_d, pose_speed)

    vis_frames.append(vis_image)

    if field_experiment:
        lc.lane_change_algo(b_dist=lead_d)


def save_video():
    if len(vis_frames) > 0:

        print('Saving video files before exit!')
        w, h = 1164, 874
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter('latest_video_processed.mp4', fourcc, 20, (w, h))

        for frame in vis_frames:
            writer.write(frame)

        writer.release()

    if len(cam_frames) > 0:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter('latest_video_raw.mp4', fourcc, 20, (1164, 874))

        for frame in cam_frames:
            writer.write(frame)

        writer.release()

        print('Video files saved!')


if __name__ == "__main__" or __name__ == "main":

    #setup_image_stream()

    try:
        # Run the pipelines as long as we have data
        while True:
            frame1, frame2 = get_frames()
            process_model(frame1, frame2)
    except BaseException as e:
        print('An exception occurred: {}'.format(e))
        save_video()
