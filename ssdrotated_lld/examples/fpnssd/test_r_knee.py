import math
import argparse
import numpy as np
import cv2
import torch
from torch.utils import data
from networks.xlsor import XLSor
from dataset.datasets import XRAYDataTestSet
import os
from PIL import Image as PILImage
from PIL import Image, ImageDraw
import torch.nn as nn

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
NUM_CLASSES = 1

DATA_DIRECTORY = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/'
DATA_LIST_PATH = 'png'
RESTORE_FROM = './examples/fpnssd/models/models_knee/knee_model.pth'


def segmentation_r_knee(name):
    try:
        DATA_DIRECTORY = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/' + name + '/right_knee/'
    
        os.environ["CUDA_VISIBLE_DEVICES"]='0'
    
        model = XLSor(num_classes=1)
        
        saved_state_dict = torch.load(RESTORE_FROM)
        model.load_state_dict(saved_state_dict)
    
        model.eval()
        model.cuda()
        
        im_list = os.listdir(DATA_DIRECTORY)
    
        for n in im_list:
            if n.endswith('png'):
                im = cv2.imread(DATA_DIRECTORY + n)
    
        h, w = im.shape[:2]
        h_w = h/512.0
        w_w = w/512.0
    
        testloader = data.DataLoader(XRAYDataTestSet(DATA_DIRECTORY,DATA_LIST_PATH, crop_size=(512, 512), mean=IMG_MEAN, scale=False, mirror=False), batch_size=1, shuffle=False, pin_memory=True)
    
        interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
    
        save_path = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/'+name+'/outputs_r_knee'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        for index, batch in enumerate(testloader):
            #if index % 100 == 0:
                #print('%d processd'%(index))
            image, size, name1 = batch
            with torch.no_grad():
                prediction = model(image.cuda(), 2)
                if isinstance(prediction, list):
                    prediction = prediction[0]
                prediction = interp(prediction).cpu().data[0].numpy().transpose(1, 2, 0)
            prediction = np.where(prediction<=0.5,0,prediction)
            prediction = np.where(prediction>1,1,prediction)
            prediction = np.where(prediction>0.5,1,prediction) 

            output_im = PILImage.fromarray((prediction[:,:,0]* 255).astype(np.uint8))
            
            rgb = PILImage.new("RGB",output_im.size)
            rgb.paste(output_im)
            rgb.save(save_path + "/" + os.path.basename(name1[0]).replace('.png', '_seg.png'), 'png')
            
            ##small object remove
            new = np.array(rgb) ## pil to cv2
            new = new[:, :, ::-1].copy()
            new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(new, connectivity=8)
            sizes = stats[1:, -1]; nb_components = nb_components - 1
            min_size = max(sizes)
            img2 = np.zeros((output.shape))
            for i in range(0, nb_components):
                #print(sizes[i])
                if sizes[i] >= min_size:
                    img2[output == i + 1] = 255
            
            cv2.imwrite(save_path + "/" + os.path.basename(name1[0]).replace('.png', '_seg_removal.png'),img2)

            im = cv2.imread(save_path + "/" + os.path.basename(name1[0]).replace('.png', '_seg_removal.png'))
            
            point_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ret, thr = cv2.threshold(point_gray, 127, 255, 0)
            _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            h, w, c = im.shape
            point_y = contours[0][0][0][1]
            y_list = []
            for i in range(0,len(contours[0])-1):
                if contours[0][i][0][0] > int(w/2) :
                    continue
                y_list.append(contours[0][i][0][1])
                if point_y <= contours[0][i][0][1]:
                    point_y = contours[0][i][0][1]
                    point_x = contours[0][i][0][0]
            
            cnt = y_list.count(point_y)
            if cnt > 1:
                point_x -= int(cnt/2)
                
            im = cv2.circle(im,(point_x,point_y),2,(0,0,255),-1)
                
            c_path = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/'+name+'/right_knee/coordinate.txt'
            f = open(c_path, 'r')
            line = f.read()
            minx = line.split(',')[0]
            miny = line.split(',')[1]
            f.close()
            
            c_path1 = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/'+name+'/right_knee/coordinate_point.txt'
            f1 = open(c_path1, 'w')

            nlx = int(minx) + w_w * point_x
            nly = int(miny) + h_w * point_y
            
            f1.write(str(int(nlx)))
            f1.write(",")
            f1.write(str(int(nly)))
            f1.close()
            
            cv2.imwrite(save_path + "/" + os.path.basename(name1[0]).replace('.png', '_xlsor.png'),im)
    except Exception as e:
        print(e)
        pass
if __name__ == '__main__':
    #name = 'aa'
    segmentation_r_knee(name)
