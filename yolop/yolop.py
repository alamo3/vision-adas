import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision
from numpy.polynomial.polynomial import polyval
from torchvision.ops import box_iou

# Model path
onnx_path = '../models_pre/yolop-640-640.onnx'

# Execution provider (Set to CUDA for now), can also be set to run on CPU
execution_providers = [
    'CUDAExecutionProvider'
]

ort_session = ort.InferenceSession(onnx_path, providers=execution_providers)

video_cap = cv2.VideoCapture('../test_video/test_creditview.mp4')


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results
    See: https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


def infer_yolop():
    """
    Runs YOLOP model on input image from video source
    :return: Image with model outputs visualized (numpy array)
    """
    global ort_session

    # read input image
    ret, img_bgr = video_cap.read()
    height, width, _ = img_bgr.shape

    # convert to RGB
    img_rgb = img_bgr[:, :, ::-1].copy()

    # resize & normalize
    canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (640, 640))

    img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
    img /= 255.0
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225

    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, 0)  # (1, 3,640,640)

    # inference: (1,n,6) (1,2,640,640) (1,2,640,640)

    # run model on input image
    det_out, da_seg_out, ll_seg_out = ort_session.run(
        ['det_out', 'drive_area_seg', 'lane_line_seg'],
        input_feed={"images": img}
    )

    # perform Non Max Suppression on detection bounding boxes
    det_out = torch.from_numpy(det_out).float()
    boxes = non_max_suppression(det_out)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
    boxes = boxes.cpu().numpy().astype(np.float32)

    # scale coords to original size.
    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, 2] -= dw
    boxes[:, 3] -= dh
    boxes[:, :4] /= r

    # select da & ll segment area.
    # da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

    # da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
    ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)

    img_merge = canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
    img_merge = img_merge[:, :, ::-1]
    img_merge = img_merge.astype(np.uint8)

    # Find contours in lane line mask to fit curves to lane lines
    contours, hierarchy = cv2.findContours(ll_seg_mask, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours that are probably just noise (less than 40 points)
    contours_filtered = []
    contour_points_x = []
    contour_points_y = []

    for i in range(len(contours)):
        if contours[i].shape[0] > 40:
            contours_filtered.append(contours[i])
            points_x = []
            points_y = []

            # If we are going to use this contour, extract its x and y coordinates for cubic curve fitting later
            for j in range(contours[i].shape[0]):
                points_x.append(contours[i][j][0][0])
                points_y.append(contours[i][j][0][1])

            contour_points_x.append(points_x)
            contour_points_y.append(points_y)

    # Now we try to fit cubic curves to the extracted lane line contour points
    contour_fits = []

    for i in range(len(contour_points_x)):
        # Perform cubic polynomial fitting on x and y arrays for each valid contour
        c, stats = np.polynomial.polynomial.polyfit(contour_points_x[i], contour_points_y[i], 3, full=True)
        contour_fits.append((c, stats))

    # Draw the output of the cubic function on the image.
    for i in range(len(contour_fits)):
        for j in range(np.amin(contour_points_x[i]), np.amax(contour_points_x[i]), 15):
            x = j
            c = contour_fits[i][0]
            y = int(polyval(j, c))

            cv2.circle(img_merge, (x, y), 2, (221, 160, 222), 1)

    # Resize image to original input size
    img_merge = cv2.resize(img_merge, (width, height), interpolation=cv2.INTER_NEAREST)

    mid_point = img_merge.shape[1] / 2
    min_from_midpoint = 1000
    bbox_lead = None

    # Draw bounding boxes for car object detection.
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        mid_bbox = (x1 + x2) / 2
        mid_bbox_y = (y1+y2 ) / 2
        from_midpoint = abs(mid_point - mid_bbox)
        if from_midpoint < min_from_midpoint:
            min_from_midpoint = from_midpoint
            bbox_lead = (x1, y1, x2, y2)

        cv2.circle(img_merge, (int(mid_bbox), int(mid_bbox_y)), 2, (0, 0, 200), 4)

        img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2), (255, 0, 0), 2, 2)

    if bbox_lead is not None:
        cv2.rectangle(img_merge, (bbox_lead[0], bbox_lead[1]), (bbox_lead[2], bbox_lead[3]), (0, 255, 0), 2, 2)

    cv2.circle(img_merge, (int(mid_point), 800), 2, (0,0,200), 4)

    return img_merge


if __name__ == "__main__":

    ort.set_default_logger_severity(4)

    # Print model inputs and outputs
    outputs_info = ort_session.get_outputs()
    inputs_info = ort_session.get_inputs()

    for ii in inputs_info:
        print("Input: ", ii)
    for oo in outputs_info:
        print("Output: ", oo)

    print("num outputs: ", len(outputs_info))

    while True:
        frame = infer_yolop()
        cv2.imshow('Detection', frame)
        cv2.waitKey(1)
