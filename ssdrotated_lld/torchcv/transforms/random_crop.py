import torch
import random


def random_crop(
        img, boxes, angles, labels,
        min_scale=0.9):
    '''Randomly crop a PIL image.

    Args:
      img: (PIL.Image) image.
      boxes: (tensor) bounding boxes, sized [#obj, 4].
      labels: (tensor) bounding box labels, sized [#obj,].
      min_scale: (float) minimal image width/height scale.

    Returns:
      img: (PIL.Image) cropped image.
      boxes: (tensor) object boxes.
      labels: (tensor) object labels.
    '''
    imw, imh = img.size
    scale_w = random.uniform(min_scale, 1)
    scale_h = random.uniform(min_scale, 1)

    w = min(int(imw * scale_w), imw - 1)
    h = min(int(imh * scale_h), imh - 1)

    x = random.randrange(imw - w)
    y = random.randrange(imh - h)

    img = img.crop((x,y,x+w,y+h))

    boxes = boxes - torch.tensor([x, y, 0, 0], dtype=torch.float)

    return img, boxes, angles, labels