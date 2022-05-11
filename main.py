import time

import onnxruntime as ort
import numpy as np
import cv2
import math

from utils import extract_preds, draw_path, Calibration, transform_img, reshape_yuv

from common.transformations.camera import eon_intrinsics
from common.transformations.model import medmodel_intrinsics

using_wide = False

providers = [
    'CPUExecutionProvider',
]

# We import our DNN model using onnx runtime and run it on the cpu
ml_session = ort.InferenceSession('supercombo.onnx', providers=providers)

# Recurrent state vector that the model uses for temporal context
state = np.zeros((1, 512)).astype(np.float32)

# Desire state vector that can be input into the model to execute turns, lane changes etc.
desire = np.zeros((1, 8)).astype(np.float32)

# open up our test file
cap = cv2.VideoCapture('test_monkey.hevc')

# average out lead information
total_dlead = 0
frame_count = 0


# This function runs the computer vision pipeline and executes the model
def run_pipeline(frame1, frame2):
    global state
    global total_dlead
    global frame_count

    orig_frame = frame1

    # Convert the frames into the YUV420 color space
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2YUV_I420)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2YUV_I420)

    # Prep the frames for the model input format
    imgs_med_model = np.zeros((2, 384, 512), dtype=np.uint8)
    imgs_med_model[0] = transform_img(frame1, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                      output_size=(512, 256))
    imgs_med_model[1] = transform_img(frame2, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                      output_size=(512, 256))

    # Convert the frame data into tensor for the model
    frame_tensors = reshape_yuv(np.array(imgs_med_model)).astype(np.float32)
    input_images = np.vstack(frame_tensors[0:2])[None]

    # We are Right hand drive traffic
    traffic_convention = np.zeros((1, 2)).astype(np.float32)
    traffic_convention[0][0] = 1

    # Input dictionary
    inputs = {'input_imgs': input_images, 'desire': desire, 'traffic_convention': traffic_convention,
              'initial_state': state}

    # Use this if using wide model (we do not use it)
    if using_wide:
        inputs = {'input_imgs': input_images, 'big_input_imgs': input_images, 'desire': desire,
                  'traffic_convention': traffic_convention,
                  'initial_state': state}

    # time it
    start = time.time()

    # finally, execute the model !!!
    pred_onnx = ml_session.run(['outputs'], inputs)

    # extract the predictions outputted by the model
    results = extract_preds(pred_onnx[0], best_plan_only=True)[0]

    lane_lines, road_edges, best_path = results

    # Get lead car information from the model output
    lead_x = pred_onnx[0][0][5755]
    lead_y = pred_onnx[0][0][5756]

    lead_d = math.sqrt(lead_x ** 2 + lead_y ** 2)
    lead_d = lead_d.__round__(2)

    # Refresh the averaging every 10 frames
    if frame_count == 10:
        total_dlead = 0
        frame_count = 0

    total_dlead = total_dlead + lead_d
    frame_count = frame_count + 1

    # Visualize the model output onto the image
    visualize(lead_d, lead_x, lead_y, orig_frame, lane_lines, road_edges, best_path)

    # print(pred_onnx[0][0][5755], pred_onnx[0][0][5756], sigmoid(pred_onnx[0][0][5857]))

    end = time.time()

    wait_time = 50 - int(((end - start) * 1000))
    if wait_time < 1:
        wait_time = 1

    # show it on the window
    cv2.waitKey(wait_time)

    # Save state recurrent vector for GRU in the next run
    state = pred_onnx[0][:, -512:]


# Some fancy math to show stuff on the image
def visualize(lead_d, lead_x, lead_y, frame, lanelines, road_edges, best_path):
    rpy_calib = [0, 0, 0]
    plot_img_height, plot_img_width = 874, 1168

    calibration_pred = Calibration(rpy_calib, plot_img_width=plot_img_width, plot_img_height=plot_img_height)

    point_lead = calibration_pred.car_space_to_bb(lead_x, lead_y, 0)

    tuple_lead = (int(point_lead[0][0]), int(point_lead[0][1]))

    radius_lead = 15 - int(0.14 * lead_d)

    if radius_lead < 5:
        radius_lead = 5

    laneline_colors = [(255, 3, 230), (0, 255, 68), (0, 255, 68), (0, 255, 68)]
    vis_image = draw_path(lanelines, road_edges, best_path[0, :, :3], frame, calibration_pred, laneline_colors,
                          width=2)

    cv2.circle(vis_image, center=tuple_lead, radius=radius_lead, color=(255, 0, 0), thickness=-1)
    cv2.putText(vis_image, 'Lead Distance = ' + str(total_dlead / frame_count), (0, 30), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 1)

    cv2.imshow('frame', vis_image)


# Sigmoid function takes in a single value
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


# Softmax function takes in a list of values
def softmax(x):
    x = np.copy(x)
    axis = 1 if len(x.shape) > 1 else 0
    x -= np.max(x, axis=axis, keepdims=True)
    if x.dtype == np.float32 or x.dtype == np.float64:
        np.exp(x, out=x)
    else:
        x = np.exp(x)
    x /= np.sum(x, axis=axis, keepdims=True)
    return x


if __name__ == "__main__":

    for input in ml_session.get_inputs():
        print(input.name + ' ' + input.type)
        print(input.shape)

    for output in ml_session.get_outputs():
        print(output.name + ' ' + output.type)

    # Run the pipelines as long as we have data
    while True:
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()

        if not (ret1 or ret2):
            break

        run_pipeline(frame1, frame2)
