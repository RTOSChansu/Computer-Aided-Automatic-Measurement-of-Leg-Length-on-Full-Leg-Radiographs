from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, num_classes):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha_list = torch.Tensor([0.25] + [0.75] * (self.num_classes-1)).to('cuda')
        self.gamma = 3

    def _focal_loss(self, x, y):
        '''Focal loss.

        This is described in the original paper.
        With BCELoss, the background should not be counted in num_classes.

        Args:
          x: (tensor) predictions, sized [N,D]., masked_cls_preds
          y: (tensor) targets, sized [N,]. cls_targets

        Return:
          (tensor) focal loss.
        '''
        logpt = F.log_softmax(x)
        logpt_ = logpt.gather(1, y.unsqueeze(1))
        logpt_ = logpt_.squeeze(1)
        pt = Variable(logpt_.data.exp())

        logpt__ = logpt * self.alpha_list
        logpt__ = logpt__.gather(1, y.unsqueeze(1))
        logpt__ = logpt__.squeeze(1)

        loss = -1 * (1 - pt) ** self.gamma * logpt__

        return loss.sum()

    #def forward(self, loc_preds, loc_targets, angle_preds, angle_targets, cls_preds, cls_targets):
    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.sum().item()

        #===============================================================
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        #===============================================================
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], size_average=False)

        #===============================================================
        # angle_loss = SmoothL1loss(pos_angle_preds, pos_angle_targets)
        #===============================================================
        #mask = pos.unsqueeze(2).expand_as(angle_preds)
        #angle_loss = F.smooth_l1_loss(angle_preds[mask], angle_targets[mask], size_average=False)

        #===============================================================
        # cls_loss = FocalLoss(cls_preds, cls_targets)
        #===============================================================
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)  # (batch_size x num_anchors) x num_classes
        cls_loss = self._focal_loss(masked_cls_preds, cls_targets[pos_neg])  # cls_targets[pos_neg] : (batch_size x num_anchors)

        if num_pos == 0:
            return None

        else:
            #print('loc_loss: %.3f | angle_loss: %.3f | cls_loss: %.3f'
                  #% (loc_loss.item()/num_pos, angle_loss.item() / num_pos, cls_loss.item()/num_pos), end=' | ')
            print('loc_loss: %.3f | cls_loss: %.3f'
                  % (loc_loss.item()/num_pos, cls_loss.item()/num_pos), end=' | ')
            #loss = (loc_loss + angle_loss + cls_loss) / num_pos
            loss = (loc_loss + cls_loss) / num_pos
            return loss
