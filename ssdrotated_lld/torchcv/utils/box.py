import cv2
import math
import numpy as np

import torch


def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax,xmax,ymax,xmax,ymax) and (xcenter,ycenter,width,height,angle).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    '''
    assert order in ['xyxy2xywh','xywh2xyxy']
    if order == 'xyxy2xywh':
        boxes_xywh = torch.zeros([boxes.size(0), 5], dtype=torch.float)
        for i in range(boxes.size(0)):
            x1 = boxes[i, 0]
            y1 = boxes[i, 1]
            x2 = boxes[i, 2]
            y2 = boxes[i, 3]
            x3 = boxes[i, 4]
            y3 = boxes[i, 5]
            x4 = boxes[i, 6]
            y4 = boxes[i, 7]
            contour = np.array([np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3]), np.array([x4, y4])])
            out = cv2.minAreaRect(contour)
            xy, wh, angle = out

            boxes_xywh[i, 0] = xy[0]
            boxes_xywh[i, 1] = xy[1]
            boxes_xywh[i, 2] = wh[0]
            boxes_xywh[i, 3] = wh[1]
            boxes_xywh[i, 4] = angle
        return boxes_xywh

    elif order == 'xywh2xyxy':
        x0 = boxes[:, 0]
        y0 = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]
        angles = boxes[:, 4] * math.pi / 180.0
        b = torch.cos(angles) * 0.5
        a = torch.sin(angles) * 0.5
        pt0_x = x0 - a * height - b * width
        pt0_y = y0 + b * height - a * width
        pt1_x = x0 + a * height - b * width
        pt1_y = y0 - b * height - a * width
        pt2_x = 2 * x0 - pt0_x
        pt2_y = 2 * y0 - pt0_y
        pt3_x = 2 * x0 - pt1_x
        pt3_y = 2 * y0 - pt1_y

        return torch.stack([pt0_x, pt0_y, pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y], 1)
    else:
        raise ValueError

def box_clamp(boxes, xmin, ymin, xmax, ymax):
    '''Clamp boxes.

    Args:
      boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [N,4].
      xmin: (number) min value of x.
      ymin: (number) min value of y.
      xmax: (number) max value of x.
      ymax: (number) max value of y.

    Returns:
      (tensor) clamped boxes.
    '''
    boxes[:,0].clamp_(min=xmin, max=xmax)
    boxes[:,1].clamp_(min=ymin, max=ymax)
    boxes[:,2].clamp_(min=xmin, max=xmax)
    boxes[:,3].clamp_(min=ymin, max=ymax)
    return boxes

def box_select(boxes, xmin, ymin, xmax, ymax):
    '''Select boxes in range (xmin,ymin,xmax,ymax).

    Args:
      boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [N,4].
      xmin: (number) min value of x.
      ymin: (number) min value of y.
      xmax: (number) max value of x.
      ymax: (number) max value of y.

    Returns:
      (tensor) selected boxes, sized [M,4].
      (tensor) selected mask, sized [N,].
    '''
    mask = (boxes[:,0]>=xmin) & (boxes[:,1]>=ymin) \
         & (boxes[:,2]<=xmax) & (boxes[:,3]<=ymax)
    boxes = boxes[mask,:]
    return boxes, mask

def box_iou(box1, box2):
    '''Compute the intersection over union of two set of rotated boxes.

    The box order must be (x_center, y_center, width, height, angle).

    Args:
      box1: (tensor) bounding boxes, sized [N,5].
      box2: (tensor) bounding boxes, sized [M,5].

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    N = box1.size(0)
    M = box2.size(0)

    inter = torch.zeros([N, M], dtype=torch.float)

    for i in range(M):
        for j in range(N):
            rn = ((box1[j, 0], box1[j, 1]), (box1[j, 2], box1[j, 3]), box1[j, 4])
            rk = ((box2[i, 0], box2[i, 1]), (box2[i, 2], box2[i, 3]), box2[i, 4])
            int_pts = cv2.rotatedRectangleIntersection(rk, rn)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area = cv2.contourArea(order_pts)
                inter[j, i] = int_area

    area1 = box1[:,2] * box1[:,3] # [N,]
    area2 = box2[:,2] * box2[:,3] # [M,]

    iou = inter / (area1[:,None] + area2 - inter)
    return iou


def box_nms(bboxes, scores, threshold=0.5):
    '''Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) confidence scores, sized [N,].
      threshold: (float) overlap threshold.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    bboxes = bboxes.numpy()
    x = bboxes[:,0]
    y = bboxes[:,1]
    w = bboxes[:,2]
    h = bboxes[:,3]
    #a = bboxes[:,4]
    areas = w * h

    _, order = scores.sort(0, descending=True)
    order = order.numpy()

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        inter = np.zeros((len(order) - 1), dtype=np.float32)

        #rn = ((x[i], y[i]), (w[i], h[i]), a[i])
        rn = ((x[i], y[i]), (w[i], h[i]),0)
        for j in range(1, len(order)):
            #rk = ((x[order[j]], y[order[j]]), (w[order[j]], h[order[j]]), a[order[j]])
            rk = ((x[order[j]], y[order[j]]), (w[order[j]], h[order[j]]),0)
            int_pts = cv2.rotatedRectangleIntersection(rk, rn)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area = cv2.contourArea(order_pts)
                inter[j-1] = int_area

        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        ids = np.where(overlap <= threshold)[0]
        if len(ids) == 0:
            break
        order = order[ids+1]
    return torch.tensor(keep, dtype=torch.long)

def py_cpu_nms(dets, scores, post_nms_topN, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0, 0]
    y1 = dets[:, 0, 1]
    x2 = dets[:, 1, 0]
    y2 = dets[:, 1, 1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ###################################################################
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ###################################################################
        ###################################################################
        # inter = np.zeros((len(order) - 1), dtype=np.float32)
        #
        # rn = ((x1[i], y1[i]), (x2[i] - x1[i], y2[i] - y1[i]), 0)
        # for j in range(1, (len(order) - 1)):
        #     rk = ((x1[order[j]], y1[order[j]]), (x2[order[j]] - x1[order[j]], y2[order[j]] - y1[order[j]]), 0)
        #     int_pts = cv2.rotatedRectangleIntersection(rk, rn)[1]
        #     if int_pts is not None:
        #         order_pts = cv2.convexHull(int_pts, returnPoints=True)
        #         int_area = cv2.contourArea(order_pts)
        #         inter[j-1] = int_area
        ###################################################################
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]

    return np.array(keep, dtype=np.int32)
