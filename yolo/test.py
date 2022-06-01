import colorsys
import math
import random
import time

import tensorflow as tf

import cv2
import numpy as np
from PIL.Image import Image
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.client.session import InteractiveSession
from tensorflow.python.saved_model import tag_constants
from model.depth_estimation import DepthEstimator

INPUT_SIZE = 416

saved_model_loaded = tf.saved_model.load('./yolov4-416', tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

video = cv2.VideoCapture('../test_video/test.hevc')

depth_estimator = DepthEstimator()


def distance(area):
    return 125.659 * (1 / (math.pow(area, 0.413185))) - 2.29504


def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape=tf.constant([416, 416])):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def draw_bbox(image, bboxes, classes=None, show_label=True):
    if classes is None:
        classes = read_class_names('coco.names')

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes

    out_boxes_image = np.zeros(shape=(num_boxes[0], 4))

    index = 0
    for box in out_boxes[0]:
        miny = box[0]*image_h
        minx = box[1]*image_w
        maxy = box[2] * image_h
        maxx = box[3] * image_w
        out_boxes_image[index] = np.array([minx, miny, maxx, maxy])
        index = index + 1
        if index == num_boxes[0]:
            break

    estimated_depths = depth_estimator.predict_depth(out_boxes_image)

    for i in range(num_boxes[0]):

        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        area = (coor[2] * 100 - coor[0] * 100) * (coor[3] * 100 - coor[1] * 100)
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])

        if not classes[class_ind] == 'car':
            continue

        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], estimated_depths[i])
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (int(c3[0]), int(c3[1])), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], int(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image


while True:

    ret, frame_orig = video.read()

    if not ret:
        break

    frame = cv2.resize(frame_orig, (INPUT_SIZE, INPUT_SIZE))
    image_data = np.asarray([frame / 255.]).astype(np.float32)

    frame_480 = cv2.resize(frame_orig, (640, 480))

    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.25
    )

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = draw_bbox(frame_480, pred_bbox)

    cv2.imshow('Detection', image)

    cv2.waitKey(1)
