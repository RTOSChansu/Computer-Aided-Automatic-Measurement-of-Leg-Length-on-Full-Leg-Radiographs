import cv2
import random
import torch
import numpy as np
from PIL import Image

from torchcv.utils.box import change_box_order


def random_rotation(img, boxes, angles, rot_factor=15):

    imw, imh = img.size
    center = (imw / 2., imh / 2.)

    angle = random.uniform(0, rot_factor)
    if random.randint(0, 1):
        angle *= -1
    rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)

    img = np.array(img)
    img = cv2.warpAffine(img, rotMat, (imw, imh))
    img = Image.fromarray(img)

    boxes_with_angles = torch.cat([boxes, angles.unsqueeze(1)], 1)
    boxes_xyxy = change_box_order(boxes_with_angles, 'xywh2xyxy')

    for i in range(boxes_xyxy.size(0)):
        for j in range(4):
            x, y = boxes_xyxy[i][j*2], boxes_xyxy[i][j*2+1]
            coor = np.array([x, y])
            R = rotMat[:, : 2]
            W = np.array([rotMat[0][2], rotMat[1][2]])
            coor = np.dot(R, coor) + W
            boxes_xyxy[i][j*2] = float(coor[0])
            boxes_xyxy[i][j*2+1] = float(coor[1])

    boxes = change_box_order(boxes_xyxy, 'xyxy2xywh')

    return img, boxes