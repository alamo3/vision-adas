import time
from datetime import datetime
from model.runnner import VisionModel
import research.lane_change as lc

import cv2
import math

# open up our test file
cap = cv2.VideoCapture('test_video/test_highway.hevc')

out_traffic = open('traffic_output.txt', "a+")

vision_model = VisionModel(using_wide=False, show_vis=True)

field_experiment = False


def log_traffic_info(lead_x, lead_y, lead_d, veh_speed):
    date_time = datetime.now()

    info = "Date: " + str(date_time) + ", Type: Lead Vehicle" + ", Distance x: " + str(lead_x) + ", Distance y: " + \
           str(lead_y) + ", Distance_t: " + str(lead_d) + " m " + ", Vehicle Speed: " + str(veh_speed * 3.6) + "\n"

    out_traffic.write(info)


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
    ret2, frame_2 = cap.read()

    if not (ret1 or ret2):
        raise Exception("Error reading from image source")

    frame_1 = res_frame(frame_1)
    frame_2 = res_frame(frame_2)

    return frame_1, frame_2


def process_model(frame1, frame2):

    global field_experiment

    lead_x, lead_y, lead_d, pose_speed = vision_model.run_model(frame1, frame2)

    log_traffic_info(lead_x, lead_y, lead_d, pose_speed)

    if field_experiment:
        lc.lane_change_algo(b_dist=lead_d, speed=pose_speed)


if __name__ == "__main__":

    setup_image_stream()

    # Run the pipelines as long as we have data
    while True:
        frame1, frame2 = get_frames()

        process_model(frame1, frame2)
