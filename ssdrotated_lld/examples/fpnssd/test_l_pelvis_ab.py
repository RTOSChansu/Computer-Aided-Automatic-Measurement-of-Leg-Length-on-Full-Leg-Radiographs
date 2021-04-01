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
import math

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
NUM_CLASSES = 1

DATA_DIRECTORY = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/'
DATA_LIST_PATH = 'png'
RESTORE_FROM = './examples/fpnssd/models/models_l_pelvis_ab/new.pth'
#RESTORE_FROM = './examples/fpnssd/models/models_l_pelvis_ab/l_pelvis_ab_model.pth'


def segmentation_l_pelvis_ab(name):
    try:
        DATA_DIRECTORY = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/' + name + '/left_pelvis_abnormal/'
        #print(DATA_DIRECTORY)
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
        
        testloader = data.DataLoader(XRAYDataTestSet(DATA_DIRECTORY, DATA_LIST_PATH, crop_size=(512, 512), mean=IMG_MEAN, scale=False, mirror=False), batch_size=1, shuffle=False, pin_memory=True)
    
        interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        
        save_path = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/'+name+'/outputs_l_pelvis'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        for index, batch in enumerate(testloader):
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
                if sizes[i] >= min_size:
                    img2[output == i + 1] = 255
            
            cv2.imwrite(save_path + "/" + os.path.basename(name1[0]).replace('.png', '_seg_removal.png'),img2)
    
            img2 = cv2.imread(save_path + "/" + os.path.basename(name1[0]).replace('.png', '_seg_removal.png'))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            rgb = PILImage.fromarray(img2)
            
            ### top point (save)
            pixels = rgb.load()
            tmp = 0
            px = 0
            py = 0
            cnt = 0
            for i in range(rgb.size[1]):
                for j in range(rgb.size[1]):
                    if pixels[j,i] == (255,255,255):
                        tmp = 1;
                        px = j
                        py = i
                        cnt = cnt + 1
                if tmp is 1:
                    break
            if cnt % 2 == 0:
                px = px - cnt/2
            else :
                px = px - cnt/2 + 1
            #pixels[px,py]=(255,0,0)
            
            draw = ImageDraw.Draw(rgb)
            draw.rectangle([(px-2,py-2),(px+2,py+2)], fill="red")
            
            c_path = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/'+name+'/left_pelvis_abnormal/coordinate.txt'
            
            f = open(c_path, 'r')
            line = f.read()
            minx = line.split(',')[0]
            miny = line.split(',')[1]
            npx = int(minx) + w_w * px
            npy = int(miny) + h_w * py
            f.close()
            c_path1 = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/'+name+'/left_pelvis_abnormal/coordinate_point.txt'
            f1 = open(c_path1, 'w')
            f1.write(str(int(npx)))
            f1.write(",")
            f1.write(str(int(npy)))
            f1.write("\n")
            
            '''### left neck point
            px_ = []
            py_ = []
            for i in range(int(px),int(0.33*rgb.size[1]),-1):
                for j in range(0,rgb.size[1],1):
                    if pixels[i,j] == (255,255,255):
                        if len(py_) >= 2 and py_[-1] > j:
                            break
                        px_.append(i)
                        py_.append(j)
                        pixels[i,j] = (0,0,255)
                        break
    
            curve = []
            tmp = 0
            lpx,lpy = 0,0
            for i in range(len(px_)-3,0,-5):
                x1 = px_[i]
                y1 = py_[i]
                x2 = px_[i+1]
                y2 = py_[i+1]
                x3 = px_[i+2]
                y3 = py_[i+2]
                
                #try:
                if y2-y1 == 0:
                    d1 = (x2-x1) / 0.1
                else: 
                    d1 = (x2-x1) / (y2-y1)
                if y3-y2 == 0:
                    d2 = (x3-x2) / 0.1
                else:
                    d2 = (x3-x2) / (y3-y2)
                if d2-d1 == 0:
                    cx = ((y3 - y1) + (x2 + x3) * d2 - (x1 + x2) * d1)/(2 * 0.1)
                else:
                    cx = ((y3 - y1) + (x2 + x3) * d2 - (x1 + x2) * d1)/(2 * (d2 - d1))
                    
                cy = -1 * d1 * (cx - (x1 + x2) / 2)+(y1 + y2)/2
                if cx > x3:
                    continue
                curvature = 1 / math.sqrt((cx-x1)**2 + (cy-y1)**2)
                #print(curvature)
                curve.append(curvature)
                if curvature > 0.17 and tmp is 0:
                    pixels[x3,y3] = (255,0,0)
                    tmpx = x2
                    tmpy = y2
                    tmp = 1
                elif curvature < 0.17:
                    lpx = x2
                    lpy = y2
                    tmp = 0
                    pixels[x3,y3] = (0,255,0)
                    #break
                    
                #except ZeroDivisionError:
                #    continue
                    
            if tmp is 0:    
                draw.rectangle([(lpx-2,lpy-2),(lpx+2,lpy+2)], fill="green")
            else:
                draw.rectangle([(tmpx-2,tmpy-2),(tmpx+2,tmpy+2)], fill="green")
    
            ### right neck point
            for i in range(rgb.size[1]-1,0,-1):
                for j in range(0,rgb.size[1],1):
                    if pixels[j,i] == (255,255,255):
                        tmpx_ = j
                        tmpy_ = i
                        break
            rpx_ = []
            rpy_ = []
            if tmp is 0:
                for i in range(lpy,int(0.66*rgb.size[1]),1):
                    for j in range(rgb.size[1]-1,0,-1):
                        if pixels[j,i] == (255,255,255):
                            rpx_.append(j)
                            rpy_.append(i)
                            break
            else:
                for i in range(tmpy,int(0.66*rgb.size[1]),1):
                    for j in range(rgb.size[1]-1,0,-1):
                        if pixels[j,i] == (255,255,255):
                            rpx_.append(j)
                            rpy_.append(i)
                            break
            
            for i in range(len(rpx_)-3,0,-1):
                x1 = rpx_[i]
                y1 = rpy_[i]
                x2 = rpx_[i+1]
                y2 = rpy_[i+1]
                x3 = rpx_[i+2]
                y3 = rpy_[i+2]
    
                if y2-y1 == 0:
                    d1 = (x2-x1) / 0.1
                else: 
                    d1 = (x2-x1) / (y2-y1)
                if y3-y2 == 0:
                    d2 = (x3-x2) / 0.1
                else:
                    d2 = (x3-x2) / (y3-y2)
                if d2-d1 == 0:
                    cx = ((y3 - y1) + (x2 + x3) * d2 - (x1 + x2) * d1)/(2 * 0.1)
                else:
                    cx = ((y3 - y1) + (x2 + x3) * d2 - (x1 + x2) * d1)/(2 * (d2 - d1))
                cy = -1 * d1 * (cx - (x1 + x2) / 2)+(y1 + y2)/2
                if cx < x3:
                    continue
                curvature = 1 / math.sqrt((cx-x1)**2 + (cy-y1)**2)
                curve.append(curvature)
                #print(curvature)
                if curvature > 0.62:
                    pixels[x3,y3] = (255,0,0)
                elif curvature > 0.17 and curvature < 0.62:
                    pixels[x3,y3] = (0,255,0)
                    rpx = x3
                    rpy = y3
                else:
                    pixels[x3,y3] = (0,0,255)
                    rpx = x3
                    rpy = y3
                    break
          
    
            draw.rectangle([(rpx-2,rpy-2),(rpx+2,rpy+2)], fill="blue")
            
            ### Center point (save)
            if tmp is 0:
                d1 = (px-lpx) / (py-lpy)
                d2 = (rpx-lpx) / (rpy-lpy)
                cx = ((rpy-py)+(lpx+rpx)*d2-(px+lpx)*d1)/(2*(d2-d1))
                cy = -1*d1*(cx-(px+lpx)/2)+(py+lpy)/2
            else:
                d1 = (px-tmpx) / (py-tmpy)
                d2 = (rpx-tmpx) / (rpy-tmpy)
                cx = ((rpy-py)+(tmpx+rpx)*d2-(px+tmpx)*d1)/(2*(d2-d1))
                cy = -1*d1*(cx-(px+tmpx)/2)+(py+tmpy)/2
            draw.rectangle([(cx-2,cy-2),(cx+2,cy+2)], fill="red")
            
            ncx = int(minx) + w_w * cx
            ncy = int(miny) + h_w * cy
            f1.write(str(int(ncx)))
            f1.write(",")
            f1.write(str(int(ncy)))
            f1.write("\n")
            
            ### 2nd top point (save)
            if tmp is 0:
                for i in range(rgb.size[1],0,-1):
                    for j in range(0,lpx):
                        if pixels[j,i] == (255,255,255):
                            s_px = j
                            s_py = i
            else:
                for i in range(rgb.size[1]-1,0,-1):
                    for j in range(0,tmpx):
                        if pixels[j,i] == (255,255,255):
                            s_px = j
                            s_py = i  
                                              
            draw = ImageDraw.Draw(rgb)
            draw.rectangle([(s_px-2,s_py-2),(s_px+2,s_py+2)], fill="red")
            
            nscx = int(minx) + w_w * s_px
            nscy = int(miny) + h_w * s_py
            
            f1.write(str(int(nscx)))
            f1.write(",")
            f1.write(str(int(nscy)))'''
            f1.close()
    
            rgb.save(save_path + "/" + os.path.basename(name1[0]).replace('.png', '_xlsor.png'), 'png')
    except Exception:
        pass
if __name__ == '__main__':
    segmentation_l_pelvis_ab(name)
