import time
from datetime import datetime

import numpy as np

from model.runnner import VisionModel
import research.lane_change as lc
import cv2
import threading
# open up our test file we can set this to be a webcam or video
cap = cv2.VideoCapture(0)
timer = time.time()
# open up traffic output file for appending new data.
out_traffic = open('traffic_output.txt', "a+")

# Instantiate an instance of the OpenPilot vision model
vision_model = VisionModel(using_wide=False, show_vis=True)

# Set this to true when conducting field experiment. It will enable lane change algorithm and GPS
field_experiment = False

# List of opencv images (numpy arrays) so we can save video when required.
vis_frames = []

cam_frames = []

ts = np.array([[1.42070485, 0.0, -30.16740088],
                  [0.0, 1.42070485, 91.030837],
                  [0.0, 0.0, 1.0]])


def log_traffic_info(lead_x, lead_y, lead_d, veh_speed):
    """
    Logs surrounding traffic info to traffic_output.txt.
    :param lead_x: Distance of lead horizontal offset from camera in image.
    :param lead_y: Distance of lead vertical offset from camera in image.
    :param lead_d: Distance of lead from camera in metres
    :param veh_speed: Vehicle speed in m/s
    :return: None
    """
    date_time = datetime.now()

    info = "Date: " + str(date_time) + ", Type: Lead Vehicle" + ", Distance x: " + str(lead_x) + ", Distance y: " + \
           str(lead_y) + ", Distance_t: " + str(lead_d) + " m " + ", Vehicle Speed: " + str(veh_speed * 3.6) + "\n"

    out_traffic.write(info)


def res_frame_2(frame):
    """
    Resizes frame to 1164 x 874 to match eon camera frame.
    :param frame: RGB image (numpy array)
    :return: resized image (numpy array)
    """
    return cv2.resize(frame, (1164, 874), interpolation=cv2.INTER_NEAREST)


def res_frame(frame):
    """
    Resizes frame to 1164 x 874 to match eon camera frame. (Zooms into frame)
    :param frame: RGB image (numpy array)
    :return: resized image (numpy array)
    """
    frame1_shape = frame.shape

    if frame1_shape[0] != 874 and frame1_shape[1] != 1168:  # check if we have to correct size
        if frame1_shape[0] >= 874 and frame1_shape[1] >= 1168:  # check if image has enough width

            crop_vertical = (frame1_shape[0] - 874) // 2  # crop equally horizontally and vertically
            crop_horizontal = (frame1_shape[1] - 1164) // 2

            return frame[crop_vertical:frame1_shape[0] - crop_vertical,  # return cropped image
                   crop_horizontal:frame1_shape[1] - crop_horizontal]
        else:
            raise Exception("This image source cannot be used for model!")

    return frame  # if image was correct size to begin with, just return it


def setup_image_stream():
    """
    Sets up video source streaming properties such as resolution and exposure
    :return: None
    """
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


def get_frames():
    """
    Returns appropriate frames for openpilot model input from defined image source.
    :return: Tuple of 2 frames (frame_1, frame_2) for input into openpilot model (numpy array)
    """
    ret1, frame_1 = cap.read()
    ret2, frame_2 = cap.read()

    if not (ret1 or ret2):
        raise Exception("Error reading from image source")

    frame_1 = res_frame_2(frame_1)  # resize frames
    frame_2 = res_frame_2(frame_2)

    cam_frames.append(frame_1)   # append to camera frames for saving video later
    cam_frames.append(frame_2)

    return frame_1, frame_2


def process_model(frame1, frame2):
    """
    Runs input frames through the openpilot model and extracts outputs from it
    :param frame1: First frame (numpy array)
    :param frame2: Second frame (numpy array)
    :return: None
    """
    global field_experiment

    # Run model
    lead_x, lead_y, lead_d, pose_speed, vis_image = vision_model.run_model(frame1, frame2)

    # Log relevant info
    log_traffic_info(lead_x, lead_y, lead_d, pose_speed)

    # Append frame with visualization to save video later
    vis_frames.append(vis_image)

    # Run lane change algo if doing field experiment
    if field_experiment:
        lc.lane_change_algo(b_dist=lead_d)


def save_video():
    """
    Saves two video files from current program execution session time stamped with date and time of saving.
    Videos/latest_video_processed_datetime.mp4 (Video with model outputs overlayed on top)
    Videos/latest_video_raw_datetime.mp4 (Direct video from camera without any processing)
    :return: None
    """

    global timer
    while True:
        '''
        if len(vis_frames) > 0:
            print('Saving video files before exit!')
            w, h = 1164, 874
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
            writer = cv2.VideoWriter('Videos/latest_video_processed_' + date +'.mp4', fourcc, 20, (w, h))

            for frame in vis_frames:
                writer.write(frame)


            writer.release()
'''

        if len(cam_frames) > 0 and timer - time.time() >= 60:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
            writer = cv2.VideoWriter('Videos/latest_video_raw_' + date + '.mp4', fourcc, 20, (1164, 874))

            for frame in cam_frames:
                writer.write(frame)
                d=cam_frames
            writer.release()
            timer = time.time()

            print('Video files saved!')
            cam_frames.clear()



if __name__ == "__main__":

    setup_image_stream()



    try:
        # Run the pipelines as long as we have data

        t2 = threading.Thread(target=save_video)
        t2.start()
        while True:
            frame1, frame2 = get_frames()
            process_model(frame1, frame2)

            # starting thread 1

            # starting thread 2

    except BaseException as e:
        print('An exception occurred: {}'.format(e))
    finally:
        save_video()  # save videos at all times.
