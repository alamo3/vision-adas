import time

import onnx
from onnx import helper

import onnxruntime as ort
import numpy as np
import cv2
import math

from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics

ml_session = ort.InferenceSession('supercombo_test.onnx')
ml_session.enable_fallback()

state = np.zeros((1, 512)).astype(np.float32)
desire = np.zeros((1, 8)).astype(np.float32)

cap = cv2.VideoCapture('sample.hevc')


def frames_to_tensor(frames):
    H = (frames.shape[1] * 2) // 3
    W = frames.shape[2]
    in_img1 = np.zeros((frames.shape[0], 6, H // 2, W // 2), dtype=np.uint8)

    in_img1[:, 0] = frames[:, 0:H:2, 0::2]
    in_img1[:, 1] = frames[:, 1:H:2, 0::2]
    in_img1[:, 2] = frames[:, 0:H:2, 1::2]
    in_img1[:, 3] = frames[:, 1:H:2, 1::2]
    in_img1[:, 4] = frames[:, H:H + H // 4].reshape((-1, H // 2, W // 2))
    in_img1[:, 5] = frames[:, H + H // 4:H + H // 2].reshape((-1, H // 2, W // 2))
    return in_img1


def process_image(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
    img_yuv = img_yuv.reshape((874 * 3 // 2, 1164))
    return transform_img(img_yuv, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                         output_size=(512, 256))


def run_pipeline(frame1, frame2):
    global state
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV_I420)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV_I420)

    imgs_med_model = np.zeros((2, 384, 512), dtype=np.uint8)
    imgs_med_model[0] = transform_img(frame1, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                      output_size=(512, 256))
    imgs_med_model[1] = transform_img(frame2, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                      output_size=(512, 256))

    cv2.imwrite('test1.png', imgs_med_model[0])
    cv2.imwrite('test2.png', imgs_med_model[1])

    frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32) / 128.0 - 1.0

    input_images = np.vstack(frame_tensors[0:2])[None]

    traffic_convention = np.zeros((1,2)).astype(np.float32)
    traffic_convention[0][0] = 1

    inputs = {'input_imgs': input_images, 'desire': desire, 'traffic_convention': traffic_convention, 'initial_state': state}

    start = time.time()

    pred_onnx = ml_session.run(['outputs', '1030'], inputs)

    #print(sigmoid(pred_onnx[0][0][585]))
    print(softmax(pred_onnx[1][0]))

    state = np.array([pred_onnx[0][0][5960:6472]]).astype(np.float32)

    end = time.time()

    #print(end-start)

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

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
        print(input.name+ ' '+input.type)
        print(input.shape)

    for output in ml_session.get_outputs():
        print(output.name + ' ' + output.type)

    while True:
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()

        if not (ret1 or ret2):
            break

        run_pipeline(frame1, frame2)
        cv2.imshow('frame', frame1)
        cv2.waitKey(45)
        cv2.imshow('frame', frame2)
