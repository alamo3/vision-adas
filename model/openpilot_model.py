import math
import time

import onnxruntime as ort
import numpy as np
import cv2

import utils
from utils import extract_preds, draw_path, Calibration, transform_img, reshape_yuv

from common.transformations.camera import eon_intrinsics, webcam_intrinsics, logitech_intrinsics
from common.transformations.camera import pretransform_from_calib
from common.transformations.model import medmodel_intrinsics

from calibration.openpilot_calib import Calibrator

providers = [
    'CPUExecutionProvider',
]


class VisionModel:

    def __init__(self, using_wide, show_vis, cam_calib=None, use_model_speed=False):

        self.using_wide = using_wide
        self.show_vis = show_vis

        if not cam_calib:
            self.cam_calib = Calibrator()
        else:
            self.cam_calib = cam_calib

        self.use_model_speed = use_model_speed

        self.use_calibration = False

        self.vehicle_speed = 0

        self.ml_session = ort.InferenceSession('models_pre/supercombo.onnx',
                                               providers=providers)

        print('Loaded model with following inputs:')

        for input in self.ml_session.get_inputs():
            print(input.name + ' ' + input.type)
            print(input.shape)

        print('Loaded model with following outputs:')

        for output in self.ml_session.get_outputs():
            print(output.name + ' ' + output.type)

        # Recurrent state vector that the model uses for temporal context
        self.state = np.zeros((1, 512)).astype(np.float32)

        # Desire state vector that can be input into the model to execute turns, lane changes etc.
        self.desire = np.zeros((1, 8)).astype(np.float32)

        # average out lead information
        self.total_dlead = 0
        self.frame_count = 0

    def run_model(self, frame1, frame2):
        orig_frame = frame1

        cv2.imshow('calibrated', frame1)

        # Convert the frames into the YUV420 color space
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV_I420)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV_I420)

        calib_rpy = self.cam_calib.get_calibration()['roll_pitch_yaw']
        pretransform = pretransform_from_calib(calib_rpy)
        calib_rpy = np.array([0, calib_rpy[1], calib_rpy[2]])

       # print(calib_rpy)

        # Prep the frames for the model input format
        imgs_med_model = np.zeros((2, 384, 512), dtype=np.uint8)
        imgs_med_model[0] = transform_img(frame1, from_intr=logitech_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                          output_size=(512, 256), augment_eulers=calib_rpy, pretransform=pretransform)
        imgs_med_model[1] = transform_img(frame2, from_intr=logitech_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                          output_size=(512, 256),
                                          augment_eulers=calib_rpy, pretransform=pretransform)

        cv2.imshow('yuv', imgs_med_model[0])

        # Convert the frame data into tensor for the model
        frame_tensors = reshape_yuv(np.array(imgs_med_model)).astype(np.float32)
        input_images = np.vstack(frame_tensors[0:2])[None]

        # We are Right hand drive traffic
        traffic_convention = np.zeros((1, 2)).astype(np.float32)
        traffic_convention[0][0] = 1

        # Input dictionary
        inputs = {'input_imgs': input_images, 'desire': self.desire, 'traffic_convention': traffic_convention,
                  'initial_state': self.state}

        # Use this if using wide model (we do not use it)
        if self.using_wide:
            inputs = {'input_imgs': input_images, 'big_input_imgs': input_images, 'desire': self.desire,
                      'traffic_convention': traffic_convention,
                      'initial_state': self.state}

        # time it
        start = time.time()

        # finally, execute the model !!!
        pred_onnx = self.ml_session.run(['outputs'], inputs)

        # extract the predictions outputted by the model
        results = extract_preds(pred_onnx[0], best_plan_only=True)[0]

        lane_lines, road_edges, best_path = results

        # Get lead car information from the model output
        lead_x = pred_onnx[0][0][5755]
        lead_y = pred_onnx[0][0][5756]
        lead_prob = pred_onnx[0][0][5857]
        lead_prob = utils.sigmoid(lead_prob)

        # print(lead_prob)

        pose_speed_x = pred_onnx[0][0][5948]
        pose_speed_y = pred_onnx[0][0][5949]
        pose_speed_z = pred_onnx[0][0][5950]
        roll = pred_onnx[0][0][5951]
        pitch = pred_onnx[0][0][5952]
        yaw = pred_onnx[0][0][5953]

        pose_speed_x_std = pred_onnx[0][0][5954]
        pose_speed_y_std = pred_onnx[0][0][5955]
        pose_speed_z_std = pred_onnx[0][0][5956]

        pose_speed = math.sqrt(pose_speed_x ** 2 + pose_speed_y ** 2 + pose_speed_z ** 2)

        if self.use_model_speed:
            self.vehicle_speed = pose_speed

        self.cam_calib.update_car_speed(self.vehicle_speed)

        lead_d = math.sqrt(lead_x ** 2 + lead_y ** 2)
        lead_d = lead_d.__round__(2)

        # Refresh the averaging every 10 frames
        if self.frame_count == 10:
            self.total_dlead = 0
            self.frame_count = 0

        self.total_dlead = self.total_dlead + lead_d
        self.frame_count = self.frame_count + 1

        # Save state recurrent vector for GRU in the next run
        self.state = pred_onnx[0][:, -512:]

        new_rpy = self.cam_calib.update_calibration_movement([pose_speed_x, pose_speed_y, pose_speed_z],
                                                             [roll, pitch, yaw],
                                                             [pose_speed_x_std, pose_speed_y_std, pose_speed_z_std],
                                                             None)

        if new_rpy is not None:
            self.use_calibration = True
            calib_result = self.cam_calib.get_calibration()
            # print('Calibration status: ', self.cam_calib.cal_status, 'Calibration: ', calib_result['cal_percentage'], [math.degrees(x) for x in new_rpy])

        # Visualize the model output onto the image

        vis_image = None

        if self.show_vis:
            vis_image = self.visualize(lead_d, lead_x, lead_y, lead_prob, orig_frame, lane_lines, road_edges, best_path)

        end = time.time()

        wait_time = 50 - int(((end - start) * 1000))
        if wait_time < 1:
            wait_time = 1

        # show it on the window
        if self.show_vis:
            cv2.waitKey(wait_time)

        return lead_x, lead_y, lead_d, pose_speed, vis_image

    # Some fancy math to show stuff on the image
    def visualize(self, lead_d, lead_x, lead_y, lead_prob, frame, lanelines, road_edges, best_path):
        calib = self.cam_calib.get_calibration()
        percent = calib["cal_percentage"]
        status = calib["cal_status"]

        calib_rpy = self.cam_calib.get_calibration()['roll_pitch_yaw']

        calib_rpy = np.array([0, calib_rpy[1], calib_rpy[2]])

        #M = utils.get_transform_matrix(base_img=frame, augment_eulers=calib_rpy, output_size=(1168, 874))

        #frame = cv2.warpPerspective(frame, M, dsize=(1168, 874))

        # rpy_calib = [0, math.radians(-0.2), math.radians(2)]
        plot_img_height, plot_img_width = 874, 1168

        calibration_pred = Calibration(calib_rpy, plot_img_width=plot_img_width,
                                       plot_img_height=plot_img_height)

        point_lead = calibration_pred.car_space_to_bb(lead_d, lead_y, 1.22)

        tuple_lead = (int(point_lead[0][0]), int(point_lead[0][1]))

        radius_lead = 15 - int(0.14 * lead_d)

        if radius_lead < 5:
            radius_lead = 5

        laneline_colors = [(255, 3, 230), (0, 255, 68), (0, 255, 68), (0, 255, 68)]
        vis_image = draw_path(lanelines, road_edges, best_path[0, :, :3], frame, calibration_pred, laneline_colors,
                              width=2)

        cv2.circle(vis_image, center=tuple_lead, radius=radius_lead, color=(255, 0, 0), thickness=-1)
        cv2.putText(vis_image, 'Lead Distance = ' + str(round(self.total_dlead / self.frame_count, 3)) + ' m', (0, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 4)
        cv2.putText(vis_image, 'Lead Probability = ' + str(round(lead_prob * 100, 1)) + ' %', (0, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 4)
        cv2.putText(vis_image, 'Calibration Percentage = ' + str(percent), (650,30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 4)
        cv2.putText(vis_image, 'Calibration Status = ' + str(status),(650,70), cv2.FONT_HERSHEY_PLAIN, 2,(0, 0, 255), 4 )
        cv2.imshow('frame', vis_image)

        return vis_image
