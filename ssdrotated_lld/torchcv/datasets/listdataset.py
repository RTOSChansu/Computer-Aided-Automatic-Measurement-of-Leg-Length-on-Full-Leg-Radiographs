from __future__ import print_function

import os

import torch
import torch.utils.data as data

from PIL import Image


class ListDataset(data.Dataset):
    '''Load image/labels/boxes from a list file.

    The list file is like:
      a.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
    '''
    def __init__(self, root, list_file, transform=None):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str/[str]) path to index file.
          transform: (function) image/box transforms.
        '''
        self.root = root
        self.transform = transform

        self.fnames = []
        self.boxes = []
        self.angles = []
        self.labels = []

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 6
            box = []
            angle = []
            label = []
            for i in range(num_boxes):
                x_center = splited[1+6*i]
                y_center = splited[2+6*i]
                width = splited[3+6*i]
                height = splited[4+6*i]
                a = splited[5+6*i]
                c = splited[6+6*i]
                box.append([float(x_center),float(y_center),float(width),float(height)])
                angle.append(float(a))
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.angles.append(torch.Tensor(angle))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          boxes: (tensor) bounding box targets.
          labels: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # use clone to avoid any potential change.
        boxes = self.boxes[idx].clone()
        angles = self.angles[idx].clone()
        labels = self.labels[idx].clone()

        if self.transform:
            img, boxes, labels = self.transform(img, boxes, angles, labels)

        boxes_ = torch.zeros((2000, 5), dtype=torch.float32)
        labels_ = torch.zeros((2000), dtype=torch.long)
        data_num = boxes.size(0)
        boxes_[:boxes.size(0),:] = boxes
        labels_[:labels.size(0)] = labels

        return img, boxes_, labels_, data_num

    def __len__(self):
        return self.num_imgs