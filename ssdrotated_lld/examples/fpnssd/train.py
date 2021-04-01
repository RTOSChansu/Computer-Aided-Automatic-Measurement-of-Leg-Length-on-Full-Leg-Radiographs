from __future__ import print_function

import argparse
import numpy as np
import cv2
import math

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter

from torchcv.datasets import ListDataset
from torchcv.loss import SSDLoss, FocalLoss
from torchcv.models.fpnssd import FPNSSD512
from torchcv.models.fpnssd import FPNSSDBoxCoder
from torchcv.transforms import resize, random_crop, random_distort, random_rotation
from sklearn.metrics import average_precision_score as ap

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class RotatedSSD:
    def __init__(self, opt):
        self.opt = opt

        # Data
        print('==> Preparing dataset..')
        self.img_size = self.opt.image_size
        self.box_coder = FPNSSDBoxCoder()

        self.dataset_train = ListDataset(root=self.opt.data_root,
                               list_file=self.opt.anno_file_list_train,
                               transform=self.transform_train)

        self.dataset_val = ListDataset(root=self.opt.data_root,
                              list_file=self.opt.anno_file_list_val,
                              transform=self.transform_val)

        self.trainloader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.opt.batch_size_train, shuffle=True, num_workers=self.opt.num_workers)
        self.valloader = torch.utils.data.DataLoader(self.dataset_val, batch_size=self.opt.batch_size_val, shuffle=False, num_workers=self.opt.num_workers)

        # Model
        print('==> Building model..')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = FPNSSD512(num_classes=self.opt.num_classes).to(self.device)
        self.init_weights()

        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

        self.best_loss = float('inf')  # best test loss
        self.start_epoch = 0  # start from epoch 0 or last epoch
        if self.opt.resume_path:
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(self.opt.resume_path)
            self.net.load_state_dict(checkpoint['net'])
            self.best_loss = checkpoint['loss']
            self.start_epoch = checkpoint['epoch']

        #self.criterion = SSDLoss(num_classes=self.opt.num_classes)
        self.criterion = FocalLoss(num_classes=self.opt.num_classes)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.opt.lr)

    def trainer(self):
        for epoch in range(self.start_epoch, self.opt.end_epoch):
            if epoch in self.opt.lr_steps:
                args.step_index += 1
                self.adjust_learning_rate(self.opt.gamma, self.opt.step_index)
            self.train(epoch)
            self.valid(epoch)

        state = {
            'net': self.net.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, os.path.join(self.opt.save_path, 'ckpt_last.pth'))

    # Training
    def train(self, epoch):
        path = './examples/fpnssd/demo/result'
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        for batch_idx, (inputs, boxes, labels, data_num) in enumerate(self.trainloader):

            for i in range(inputs.size(0)):
                boxes_sample = boxes[i, :int(data_num[i]), :]
                labels_sample = labels[i, :int(data_num[i])]

                boxes_sample, angles_sample, labels_sample, anchor_boxes_sample = self.box_coder.encode(boxes_sample, labels_sample)
                
                # ### For Debug ##########################################################################################
                # variances = (0.1, 0.2)
                # xy = boxes_sample[:, :2] * variances[0] * anchor_boxes_sample[:, 2:4] + anchor_boxes_sample[:, :2]
                # wh = (boxes_sample[:, 2:4] * variances[1]).exp() * anchor_boxes_sample[:, 2:4]
                # loc_angle = (angles_sample[:, 0]) + anchor_boxes_sample[:, 4]
                #
                # image = inputs[i].permute(1, 2, 0).numpy()
                # im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # box_preds = torch.cat([xy, wh, loc_angle.unsqueeze(1)], 1)
                # pos_indices = (labels_sample != 0).nonzero()
                #
                # # for i in range(len(anchor_boxes_sample)):
                # #     im_ = im.copy()
                # #     # if i < 35000:
                # #     #     continue
                # #     box = anchor_boxes_sample[i]
                # #     box_points = cv2.boxPoints(((float(box[0]),float(box[1])), (float(box[2]), float(box[3])), float(box[4])))
                # #     box_points = np.int0(box_points)
                # #     cv2.drawContours(im_, [box_points], 0, [255,0,0], 2)
                # #     cv2.putText(im_, text='{0:.4f}'.format(box[4]), org=(int(box_points[0, 0]), int(box_points[0, 1])),
                # #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255,0,0))
                # #     cv2.imshow('image', im_)
                # #     cv2.waitKey(0)
                #
                # for pos_index in pos_indices:
                #     pos_index = int(pos_index)
                #     box = box_preds[pos_index]
                #     box_points = cv2.boxPoints(
                #         ((float(box[0]), float(box[1])), (float(box[2]), float(box[3])), float(box[4])))
                #     box_points = np.int0(box_points)
                #     cv2.drawContours(im, [box_points], 0, [255, 0, 0], 2)
                #
                #     box_a = anchor_boxes_sample[pos_index]
                #     box_points_a = cv2.boxPoints(
                #         ((float(box_a[0]), float(box_a[1])), (float(box_a[2]), float(box_a[3])), float(box_a[4])))
                #     box_points_a = np.int0(box_points_a)
                #     cv2.drawContours(im, [box_points_a], 0, [0, 0, 255], 2)
                #
                #     cv2.imshow('image', im)
                #     cv2.waitKey(0)

                if i == 0:
                    loc_targets = boxes_sample.unsqueeze(0)
                    angle_targets = angles_sample.unsqueeze(0)
                    cls_targets = labels_sample.unsqueeze(0)
                else:
                    loc_targets = torch.cat([loc_targets, boxes_sample.unsqueeze(0)], 0)
                    angle_targets = torch.cat([angle_targets, angles_sample.unsqueeze(0)], 0)
                    cls_targets = torch.cat([cls_targets, labels_sample.unsqueeze(0)], 0)

            inputs = inputs.to(self.device)
            loc_targets = loc_targets.to(self.device)
            angle_targets = angle_targets.to(self.device)
            cls_targets = cls_targets.to(self.device)

            #loc_preds, angle_preds, cls_preds = self.net(inputs)
            loc_preds, cls_preds = self.net(inputs)

            #loss = self.criterion(loc_preds, loc_targets, angle_preds, angle_targets, cls_preds, cls_targets)
            loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
                  % (loss.item(), train_loss / (batch_idx + 1), batch_idx + 1, len(self.trainloader)))

    def valid(self, epoch):
        print('\nTest')
        self.net.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, boxes, labels, data_num) in enumerate(self.valloader):
                for i in range(inputs.size(0)):
                    boxes_sample = boxes[i, :int(data_num[i]), :]
                    labels_sample = labels[i, :int(data_num[i])]

                    boxes_sample, angles_sample, labels_sample, anchor_boxes_sample = self.box_coder.encode(boxes_sample,
                                                                                                       labels_sample)
                    if i == 0:
                        loc_targets = boxes_sample.unsqueeze(0)
                        angle_targets = angles_sample.unsqueeze(0)
                        cls_targets = labels_sample.unsqueeze(0)
                    else:
                        loc_targets = torch.cat([loc_targets, boxes_sample.unsqueeze(0)], 0)
                        angle_targets = torch.cat([angle_targets, angles_sample.unsqueeze(0)], 0)
                        cls_targets = torch.cat([cls_targets, labels_sample.unsqueeze(0)], 0)

                inputs = inputs.to(self.device)
                loc_targets = loc_targets.to(self.device)
                angle_targets = angle_targets.to(self.device)
                cls_targets = cls_targets.to(self.device)

                #loc_preds, angle_preds, cls_preds = self.net(inputs)
                loc_preds, cls_preds = self.net(inputs)
                #loss = self.criterion(loc_preds, loc_targets, angle_preds, angle_targets, cls_preds, cls_targets)
                loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
                #print("target : "+cls_targets+', predicted:'+cls_preds)


                # if loss is None:
                #     continue
                valid_loss += loss.item()
                print('test_loss: %.3f | avg_loss: %.3f [%d/%d]'
                      % (loss.item(), valid_loss / (batch_idx + 1), batch_idx + 1, len(self.valloader)))

        # Save checkpoint
        valid_loss /= len(self.valloader)
        if valid_loss < self.best_loss:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'loss': valid_loss,
                'epoch': epoch,
            }
            if not os.path.isdir(os.path.dirname(self.opt.save_path)):
                os.mkdir(os.path.dirname(self.opt.save_path))
            torch.save(state, os.path.join(self.opt.save_path, 'ckpt_best.pth'))
            self.best_loss = valid_loss

    def transform_train(self, img, boxes, angles, labels):
        #img = random_distort(img)
        #img, boxes, angles, labels = random_crop(img, boxes, angles, labels)
        #img, boxes = random_rotation(img, boxes, angles)

        boxes = torch.cat([boxes, angles.unsqueeze(1)], 1)
        img, boxes = resize(img, boxes, size=(self.img_size, self.img_size), random_interpolation=True)

        # ### for debug ##############################################################
        # boxes_ = boxes.numpy()
        # im = np.array(img)
        # for box in boxes_:
        #     box_points = cv2.boxPoints(((box[0],box[1]), (box[2], box[3]), box[4]))
        #     box_points = np.int0(box_points)
        #     cv2.drawContours(im, [box_points], 0, [255,0,0], 2)
        # cv2.imshow('image', im)
        # cv2.waitKey(0)
        # ############################################################################

        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])(img)

        return img, boxes, labels

    def transform_val(self, img, boxes, angles, labels):
        boxes = torch.cat([boxes, angles.unsqueeze(1)], 1)
        img, boxes = resize(img, boxes, size=(self.img_size, self.img_size))
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])(img)
        return img, boxes, labels

    def init_weights(self):
        state_dict = torch.load(self.opt.initial_path)
        own_state = self.net.state_dict()
        for name, param in state_dict.items():
            name = name.replace('module.', '')
            if name in own_state:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)

                except:
                    print('While copying the parameter named {}, '
                          'whose dimensions in the model are {} and '
                          'whose dimensions in the checkpoint are {}.'
                          .format(name, own_state[name].size(), param.size()))
            else:
                print('Unexpected key(s) in state_dict: {}'.format(name))
        print("loading weights %s" % self.opt.initial_path)

    def adjust_learning_rate(self, gamma, step):
        """Sets the learning rate to the initial LR decayed by 10 at every
            specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        lr = args.lr * (gamma ** (step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch RotatedSSD Training')
    parser.add_argument('--image_size',           type=int,  default=512,                            help='Input image size')
    parser.add_argument('--num_classes',          type=int,  default=13,                              help='Number of classes')
    parser.add_argument('--batch_size_train',     type=int,  default=10,                             help='Batch size for training')
    parser.add_argument('--batch_size_val',       type=int,  default=10,                             help='Batch size for validation')
    parser.add_argument('--num_workers',           type=int,  default=4,                              help='Number of workers')

    parser.add_argument('--data_root',            type=str,  default='./data/ba',                help='Data root path')
    parser.add_argument('--anno_file_list_train', type=list, default=['./data/ba/BA_train.txt'],   help='Anno file list for train')
    parser.add_argument('--anno_file_list_val',   type=list, default=['./data/ba/BA_val.txt'],   help='Anno file list for valid')
    parser.add_argument('--initial_path',         type=str,  default='./examples/fpnssd/pretrained/fpnssd512_20.pth',  help='Initial path')
    #parser.add_argument('--initial_path',         type=str,  default='./examples/fpnssd/checkpoint/ckpt_best.pth',  help='Initial path')
    parser.add_argument('--resume_path',          type=str,  default=None,                           help='Resume path')
    parser.add_argument('--save_path',            type=str,  default='./examples/fpnssd/checkpoint/new/',                   help='Save path')

    parser.add_argument('--gamma',                type=float, default=0.1,                           help='Gamma update for SGD')
    parser.add_argument('--lr_steps',             type=list,  default=[10, 20, 30, 40, 50],                help='Gamma update for SGD')
    parser.add_argument('--step_index',           type=int,   default=0,                             help='Gamma update for SGD')
    parser.add_argument('--lr',                   type=float, default=1e-3,                          help='Learning rate')
    parser.add_argument('--end_epoch',            type=int,   default=50,                          help='End epoch')

    args = parser.parse_args()

    RotatedSSD_ = RotatedSSD(args)
    RotatedSSD_.trainer()
