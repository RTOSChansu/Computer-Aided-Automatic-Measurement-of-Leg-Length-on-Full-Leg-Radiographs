import argparse
import numpy as np
import math
import torch
from torch.utils import data
from networks.xlsor import XLSor
from dataset.datasets import XRAYDataTestSet
import os
from PIL import Image as PILImage
from PIL import Image, ImageDraw
import torch.nn as nn
import cv2

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
NUM_CLASSES = 1

DATA_DIRECTORY = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/'
DATA_LIST_PATH = 'png'
RESTORE_FROM = './examples/fpnssd/models/models_l_ankle_ab/l_ankle_ab_model.pth'

def segmentation_l_ankle_ab(name):
    try:

        DATA_DIRECTORY = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/' + name + '/left_ankle_abnormal/'
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
    
        save_path = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/'+name+'/outputs_l_ankle'
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

            img2 = cv2.imread(save_path + "/" + os.path.basename(name1[0]).replace('.png', '_seg_removal.png'))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            rgb = PILImage.fromarray(img2)
            
            '''point detection'''
            pixels = rgb.load()
            px = []
            py = []
            px_ = []
            py_ = []
            
            ## left top point
            for i in range(int(rgb.size[1]/2)-10,1,-3):
                for j in range(0,rgb.size[1]-1,1):
                    if pixels[i,j] == (255,255,255):
                        px.append(i)
                        py.append(j)
                        #pixels[j,i] = (255,0,0)
                        break
            #print(px,py)
            for i in range(len(px)-3):
                x1 = px[i]
                y1 = py[i]
                x2 = px[i+1]
                y2 = py[i+1]
                x3 = px[i+2]
                y3 = py[i+2]

                if int(x3-x1) is 0 and int(x2-x1) is not 0:
                    theta = math.atan((y3-y1)/0.1) - math.atan((y2-y1)/(x2-x1))
                elif int(x2-x1) is 0 and int(x3-x1) is not 0:
                    theta = math.atan((y3-y1)/(x3-x1)) - math.atan((y2-y1)/(0.1))
                elif int(x3-x1) is 0 and int(x2-x1) is 0:
                    theta = math.atan((y3-y1)/0.1) - math.atan((y2-y1)/(0.1)) 
                else:
                    theta = math.atan((y3-y1)/(x3-x1)) - math.atan((y2-y1)/(x2-x1))
                theta = theta * 180/math.pi
                if y1 >= y3:
                    continue
                #print(theta)
                if theta >= 0 :
                    pixels[x2,y2] = (255,0,0)
                    p_ = x1
                    p__ = y1
                    if y3 - y1 > 5:
                        break
         
            draw = ImageDraw.Draw(rgb)
            draw.rectangle([(p_-2,p__-2),(p_+2,p__+2)], fill="red")
            
            ## right top point
            for i in range(int(rgb.size[1]/2)+10,int(rgb.size[1]-1),3):
                for j in range(0,int(rgb.size[1])-1,1):
                    if pixels[i,j] == (255,255,255):
                        px_.append(i)
                        py_.append(j)
                        #pixels[j,i] = (255,0,0)
                        break
                        
            for i in range(len(px_)):
                if i is len(px_)-3:
                    break
                x1 = px_[i]
                y1 = py_[i]
                x2 = px_[i+1]
                y2 = py_[i+1]
                x3 = px_[i+2]
                y3 = py_[i+2]
                if int(x3-x1) is 0 and int(x2-x1) is not 0:
                    theta = math.atan((y3-y1)/0.1) - math.atan((y2-y1)/(x2-x1))
                elif int(x2-x1) is 0 and int(x3-x1) is not 0:
                    theta = math.atan((y3-y1)/(x3-x1)) - math.atan((y2-y1)/(0.1))
                elif int(x3-x1) is 0 and int(x2-x1) is 0:
                    theta = math.atan((y3-y1)/0.1) - math.atan((y2-y1)/(0.1)) 
                else:
                    theta = math.atan((y3-y1)/(x3-x1)) - math.atan((y2-y1)/(x2-x1))
                theta = theta * 180/math.pi
                
                if y1 >= y3:
                    continue
                #print(theta)
                if theta <= 0 :
                    pixels[x2,y2] = (255,0,0)
                    rp_ = x1
                    rp__ = y1
                    if y3 - y1 > 5:
                        #print('aa')
                        break
         
            draw = ImageDraw.Draw(rgb)
            draw.rectangle([(rp_-2,rp__-2),(rp_+2,rp__+2)], fill="blue")
            
            ## Center point (save)
            cpx = int((rp_ + p_) / 2)
            for i in range(0,rgb.size[1],1):
                if pixels[cpx,i] == (255,255,255):
                    cpy = i
                    break
            draw = ImageDraw.Draw(rgb)
            draw.rectangle([(cpx-2,cpy-2),(cpx+2,cpy+2)], fill="green")
            c_path = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/'+name+'/left_ankle_abnormal/coordinate.txt'
            f = open(c_path, 'r')
            line = f.read()
            minx = line.split(',')[0]
            miny = line.split(',')[1]
            ncx = int(minx) + w_w * cpx
            ncy = int(miny) + h_w * cpy
            f.close()
            c_path1 = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/'+name+'/left_ankle_abnormal/coordinate_point.txt'
            f1 = open(c_path1, 'w')
            f1.write(str(int(ncx)))
            f1.write(",")
            f1.write(str(int(ncy)))
            f1.write("\n")
            f1.close()
            
            rgb.save(save_path + "/" + os.path.basename(name1[0]).replace('.png', '_xlsor.png'), 'png')
    except Exception:
        pass

if __name__ == '__main__':
    segmentation_l_ankle_ab(name)
