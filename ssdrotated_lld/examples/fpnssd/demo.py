import os
#os.environ["CUDA_VISIBLE_DEVICES"]='3'
import cv2
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchcv.models.fpnssd import FPNSSD512, FPNSSDBoxCoder
from test_model import start_segmentation_engine
import shutil
import time

def his(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(im)
    return eq
    
save_path = "/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result"

def demo(opt):
    
    print('Loading model..')

    net = FPNSSD512(num_classes=opt.num_classes).to('cuda')
    net = torch.nn.DataParallel(net)

    checkpoint = torch.load(opt.load_path)
    net.load_state_dict(checkpoint['net'])
    net.eval()

    image_path = opt.demo_path
    image_list = os.listdir(image_path)
 
    cnt = 0
    for image_name in image_list:
        start_time = time.time()
        if os.path.isdir(image_path + '/' + image_name):
            continue
        cnt += 1
        print('Loading image..')
        print(image_name)
        sav_dir = opt.result_path + '/' + image_name[:-4]
        if not os.path.isdir(sav_dir):
            os.makedirs(sav_dir)
        else:
            continue
        im = cv2.imread(os.path.join(image_path, image_name))
        cv2.imwrite(os.path.join(sav_dir, image_name[:-4]+'_origin.jpg'), im)
        im2 = im.copy()
        h, w = im.shape[:2]
        h_w = h/512.0
        w_w = w/512.0
        im = cv2.resize(im, (opt.image_size, opt.image_size))

        print('Predicting..')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
        x = transform(im).to('cuda')
        with torch.no_grad():
            loc_preds, cls_preds = net(x.unsqueeze(0))

        print('Decoding..')
        box_coder = FPNSSDBoxCoder()
        loc_preds = loc_preds.squeeze().cpu()
        cls_preds = F.softmax(cls_preds.squeeze(), dim=1).cpu()

        boxes, labels, scores = box_coder.decode(loc_preds, cls_preds, score_thresh=opt.score_thresh, nms_thresh=0)
        #print(boxes,labels)
        #print(h_w,w_w)
        '''for i, box in enumerate(boxes):
            if box[2] < box[3]:
                box_points = cv2.boxPoints(((float(box[0]), float(box[1])), (float(box[2]), float(box[3])), 90.))
            else:
                box_points = cv2.boxPoints(((float(box[0]), float(box[1])), (float(box[3]), float(box[2])), 90.))
            box_points = np.int0(box_points)
            cv2.drawContours(im, [box_points], 0, [255,0,0], 2)
        cv2.imwrite(os.path.join(sav_dir, image_name[:-4]+'_origin_oo.jpg'), im)'''
        #if len(boxes) < 6:
        #    break
        clss = []
        label_list = labels.tolist()
        print([labels,scores])

        if len(boxes) > 6:
            for i in range(0,len(label_list)):
                if label_list.count(label_list[i]) >= 2:
                    clss.append(i)
            if len(clss) is 0:
                chk = 0
            else:
                tp = min(scores[clss[0]],scores[clss[1]])
                chk = 1
        else:
            chk = 0

        for i, box in enumerate(boxes):
            if chk is 1:
                if scores[i].item() == tp.item():
                    continue
            tmp = 0
            box_points = cv2.boxPoints(((float(box[0]*w_w),float(box[1]*h_w)), (float(box[2]*h_w), float(box[3]*w_w)), 90.))
            #box_points = cv2.boxPoints(((float(box[0]),float(box[1])), (float(box[2]), float(box[3])), 90.))
            box_points = np.int0(box_points)
            minx = box[0]*w_w - box[3]*w_w/2
            maxx = box[0]*w_w + box[3]*w_w/2
            miny = box[1]*h_w - box[2]*h_w/2
            maxy = box[1]*h_w + box[2]*h_w/2
            if minx.item() < 0 :
                tmp = 1
                minx = 0
            if maxx.item() > w :
                tmp = 1
                maxx = w
            if miny.item() < 0 :
                tmp = 1
                miny = 0
            if maxy.item() > h :
                tmp = 1
                maxy = h
            #print(minx,miny,maxx,maxy)
            
            if labels[i].item() is 0:
                if tmp is not 1:
                    if minx.item() > w*0.66:
                        continue
                else:
                    if minx > w*0.66:
                        continue
                if os.path.isfile(sav_dir+'/left_pelvis/coordinate.txt'):
                    f = open(sav_dir+'/left_pelvis/coordinate.txt')
                    s = f.readline()
                    x = s.split(',')[0]
                    if int(x) < minx.item():
                        os.remove(sav_dir+'/left_pelvis/coordinate.txt')
                        os.remove(sav_dir+'/left_pelvis/'+image_name[:-4]+'_left.png')
                    else:
                        continue
                if not os.path.isdir(sav_dir+'/left_pelvis'):
                     os.makedirs(sav_dir+'/left_pelvis')
                left_pelvis = im2[int(miny):int(maxy),int(minx):int(maxx)]
                #left_pelvis = his(left_pelvis)
                cv2.imwrite(os.path.join(sav_dir+'/left_pelvis', image_name[:-4]+'_left.png'), left_pelvis)
                save_coordinate(sav_dir+'/left_pelvis/coordinate.txt',minx,miny)
                
            elif labels[i].item() is 1:
                if tmp is not 1:
                    if minx.item() < w*0.33:
                        continue
                else:
                    if minx < w*0.33:
                        continue
                if os.path.isfile(sav_dir+'/right_pelvis/coordinate.txt'):
                    f = open(sav_dir+'/right_pelvis/coordinate.txt')
                    s = f.readline()
                    x = s.split(',')[0]
                    if int(x) > minx.item():
                        os.remove(sav_dir+'/right_pelvis/coordinate.txt')
                        os.remove(sav_dir+'/right_pelvis/'+image_name[:-4]+'_right.png')
                    else:
                        continue
                if not os.path.isdir(sav_dir+'/right_pelvis'):
                     os.makedirs(sav_dir+'/right_pelvis')
                right_pelvis = im2[int(miny):int(maxy),int(minx):int(maxx)]
                #right_pelvis = his(right_pelvis)
                cv2.imwrite(os.path.join(sav_dir+'/right_pelvis', image_name[:-4]+'_right.png'), right_pelvis)
                save_coordinate(sav_dir+'/right_pelvis/coordinate.txt',minx,miny)
                
            elif labels[i].item() is 2:
                if tmp is not 1:
                    if minx.item() > w*0.66:
                        continue
                else:
                    if minx > w*0.66:
                        continue
                if os.path.isfile(sav_dir+'/left_knee/coordinate.txt'):
                    f = open(sav_dir+'/left_knee/coordinate.txt')
                    s = f.readline()
                    x = s.split(',')[0]
                    if int(x) > minx.item():
                        os.remove(sav_dir+'/left_knee/coordinate.txt')
                        os.remove(sav_dir+'/left_knee/'+image_name[:-4]+'_left.png')
                    else:
                        continue
                if not os.path.isdir(sav_dir+'/left_knee'):
                     os.makedirs(sav_dir+'/left_knee')
                left_knee = im2[int(miny):int(maxy),int(minx):int(maxx)]
                #left_knee = his(left_knee)
                cv2.imwrite(os.path.join(sav_dir+'/left_knee', image_name[:-4]+'_left.png'), left_knee)
                save_coordinate(sav_dir+'/left_knee/coordinate.txt',minx,miny)
                
            elif labels[i].item() is 3:
                if tmp is not 1:
                    if minx.item() < w*0.33:
                        continue
                else:
                    if minx < w*0.33:
                        continue
                if os.path.isfile(sav_dir+'/right_knee/coordinate.txt'):
                    f = open(sav_dir+'/right_knee/coordinate.txt')
                    s = f.readline()
                    x = s.split(',')[0]
                    if int(x) < minx.item():
                        os.remove(sav_dir+'/right_knee/coordinate.txt')
                        os.remove(sav_dir+'/right_knee/'+image_name[:-4]+'_right.png')
                    else:
                        continue
                if not os.path.isdir(sav_dir+'/right_knee'):
                     os.makedirs(sav_dir+'/right_knee')
                right_knee = im2[int(miny):int(maxy),int(minx):int(maxx)]
                #right_knee = his(right_knee)
                cv2.imwrite(os.path.join(sav_dir+'/right_knee', image_name[:-4]+'_right.png'), right_knee)
                save_coordinate(sav_dir+'/right_knee/coordinate.txt',minx,miny)
                
            elif labels[i].item() is 4:
                if tmp is not 1:
                    if minx.item() > w*0.66:
                        continue
                else:
                    if minx > w*0.66:
                        continue
                if os.path.isfile(sav_dir+'/left_ankle/coordinate.txt'):
                    f = open(sav_dir+'/left_ankle/coordinate.txt')
                    s = f.readline()
                    x = s.split(',')[0]
                    y = s.split(',')[1]
                    #print(x,minx.item(),y,miny.item())
                    if int(x) < minx.item() and int(y) < miny.item():
                        os.remove(sav_dir+'/left_ankle/coordinate.txt')
                        os.remove(sav_dir+'/left_ankle/'+image_name[:-4]+'_left.png')
                    else:
                        continue
                if not os.path.isdir(sav_dir+'/left_ankle'):
                     os.makedirs(sav_dir+'/left_ankle')
                left_ankle = im2[int(miny):int(maxy),int(minx):int(maxx)]
                #left_ankle = his(left_ankle)
                cv2.imwrite(os.path.join(sav_dir+'/left_ankle', image_name[:-4]+'_left.png'), left_ankle)
                save_coordinate(sav_dir+'/left_ankle/coordinate.txt',minx,miny)
                
            elif labels[i].item() is 5:
                if tmp is not 1:
                    if minx.item() < w*0.33:
                        continue
                else:
                    if minx < w*0.33:
                        continue
                if os.path.isfile(sav_dir+'/right_ankle/coordinate.txt'):
                    f = open(sav_dir+'/right_ankle/coordinate.txt')
                    s = f.readline()
                    x = s.split(',')[0]
                    y = s.split(',')[1]
                    if int(y) < miny.item():
                        os.remove(sav_dir+'/right_ankle/coordinate.txt')
                        os.remove(sav_dir+'/right_ankle/'+image_name[:-4]+'_right.png')
                    else:
                        continue
                if not os.path.isdir(sav_dir+'/right_ankle'):
                     os.makedirs(sav_dir+'/right_ankle')
                right_ankle = im2[int(miny):int(maxy),int(minx):int(maxx)]
                #right_ankle = his(right_ankle)
                #print(right_ankle)
                cv2.imwrite(os.path.join(sav_dir+'/right_ankle', image_name[:-4]+'_right.png'), right_ankle)
                save_coordinate(sav_dir+'/right_ankle/coordinate.txt',minx,miny)
                
            elif labels[i].item() is 6:
                if tmp is not 1:
                    if minx.item() > w*0.66:
                        continue
                else:
                    if minx > w*0.66:
                        continue
                if os.path.isdir(sav_dir+'/left_pelvis'):
                    l = labels.tolist()
                    if scores[i] > scores[l.index(0)]:
                        shutil.rmtree(sav_dir+'/left_pelvis')
                    else:
                        continue
                if os.path.isfile(sav_dir+'/left_pelvis_abnormal/coordinate.txt'):
                    f = open(sav_dir+'/left_pelvis_abnormal/coordinate.txt')
                    s = f.readline()
                    x = s.split(',')[0]
                    if int(x) < minx.item():
                        os.remove(sav_dir+'/left_pelvis_abnormal/coordinate.txt')
                        os.remove(sav_dir+'/left_pelvis_abnormal/'+image_name[:-4]+'_left.png')
                    else:
                        continue
                
                if not os.path.isdir(sav_dir+'/left_pelvis_abnormal'):
                     os.makedirs(sav_dir+'/left_pelvis_abnormal')
                left_pel = im2[int(miny):int(maxy),int(minx):int(maxx)]
                #left_pel = his(left_pel)
                cv2.imwrite(os.path.join(sav_dir+'/left_pelvis_abnormal', image_name[:-4]+'_left.png'), left_pel)
                save_coordinate(sav_dir+'/left_pelvis_abnormal/coordinate.txt',minx,miny)
                
            elif labels[i].item() is 7:
                if tmp is not 1:
                    if minx.item() < w*0.33:
                        continue
                else:
                    if minx < w*0.33:
                        continue
                if os.path.isdir(sav_dir+'/right_pelvis'):
                    l = labels.tolist()
                    if scores[i] > scores[l.index(1)]:
                        shutil.rmtree(sav_dir+'/right_pelvis')
                    else:
                        continue
                if os.path.isfile(sav_dir+'/right_pelvis_abnormal/coordinate.txt'):
                    f = open(sav_dir+'/right_pelvis_abnormal/coordinate.txt')
                    s = f.readline()
                    x = s.split(',')[0]
                    if int(x) > minx.item():
                        os.remove(sav_dir+'/right_pelvis_abnormal/coordinate.txt')
                        os.remove(sav_dir+'/right_pelvis_abnormal/'+image_name[:-4]+'_right.png')
                    else:
                        continue

                if not os.path.isdir(sav_dir+'/right_pelvis_abnormal'):
                     os.makedirs(sav_dir+'/right_pelvis_abnormal')
                right_pel = im2[int(miny):int(maxy),int(minx):int(maxx)]
                #right_pel = his(right_pel)
                cv2.imwrite(os.path.join(sav_dir+'/right_pelvis_abnormal', image_name[:-4]+'_right.png'), right_pel)
                save_coordinate(sav_dir+'/right_pelvis_abnormal/coordinate.txt',minx,miny)
                
            elif labels[i].item() is 8:
                if tmp is not 1:
                    if minx.item() > w*0.66:
                        continue
                else:
                    if minx > w*0.66:
                        continue
                if os.path.isdir(sav_dir+'/left_knee'):
                    l = labels.tolist()
                    if scores[i] > scores[l.index(2)]:
                        shutil.rmtree(sav_dir+'/left_knee')
                    else:
                        continue
                if os.path.isfile(sav_dir+'/left_knee_abnormal/coordinate.txt'):
                    f = open(sav_dir+'/left_knee_abnormal/coordinate.txt')
                    s = f.readline()
                    x = s.split(',')[0]
                    if int(x) > minx.item():
                        os.remove(sav_dir+'/left_knee_abnormal/coordinate.txt')
                        os.remove(sav_dir+'/left_knee_abnormal/'+image_name[:-4]+'_left.png')
                    else:
                        continue

                if not os.path.isdir(sav_dir+'/left_knee_abnormal'):
                     os.makedirs(sav_dir+'/left_knee_abnormal')
                left_knee = im2[int(miny):int(maxy),int(minx):int(maxx)]
                #left_knee = his(left_knee)
                cv2.imwrite(os.path.join(sav_dir+'/left_knee_abnormal', image_name[:-4]+'_left.png'), left_knee)
                save_coordinate(sav_dir+'/left_knee_abnormal/coordinate.txt',minx,miny)
                
            elif labels[i].item() is 9:
                if tmp is not 1:
                    if minx.item() < w*0.33:
                        continue
                else:
                    if minx < w*0.33:
                        continue
                if os.path.isdir(sav_dir+'/right_knee'):
                    l = labels.tolist()
                    if scores[i] > scores[l.index(3)]:
                        shutil.rmtree(sav_dir+'/right_knee')
                    else:
                        continue
                if os.path.isfile(sav_dir+'/right_knee_abnormal/coordinate.txt'):
                    f = open(sav_dir+'/right_knee_abnormal/coordinate.txt')
                    s = f.readline()
                    x = s.split(',')[0]
                    if int(x) < minx.item():
                        os.remove(sav_dir+'/right_knee_abnormal/coordinate.txt')
                        os.remove(sav_dir+'/right_knee_abnormal/'+image_name[:-4]+'_right.png')
                    else:
                        continue
                
                if not os.path.isdir(sav_dir+'/right_knee_abnormal'):
                     os.makedirs(sav_dir+'/right_knee_abnormal')
                right_knee = im2[int(miny):int(maxy),int(minx):int(maxx)]
                #right_knee = his(right_knee)
                cv2.imwrite(os.path.join(sav_dir+'/right_knee_abnormal', image_name[:-4]+'_right.png'), right_knee)
                save_coordinate(sav_dir+'/right_knee_abnormal/coordinate.txt',minx,miny)
    
            elif labels[i].item() is 10:
                if tmp is not 1:
                    if minx.item() > w*0.66:
                        continue
                else:
                    if minx > w*0.66:
                        continue
                if os.path.isdir(sav_dir+'/left_ankle'):
                    l = labels.tolist()
                    if scores[i] > scores[l.index(4)]:
                        shutil.rmtree(sav_dir+'/left_ankle')
                    else:
                        continue
                if os.path.isfile(sav_dir+'/left_ankle_abnormal/coordinate.txt'):
                    f = open(sav_dir+'/left_ankle/coordinate.txt')
                    s = f.readline()
                    x = s.split(',')[0]
                    if int(x) < minx.item() and int(y) < miny.item():
                        os.remove(sav_dir+'/left_ankle_abnormal/coordinate.txt')
                        os.remove(sav_dir+'/left_ankle_abnormal/'+image_name[:-4]+'_left.png')
                    else:
                        continue

                if not os.path.isdir(sav_dir+'/left_ankle_abnormal'):
                     os.makedirs(sav_dir+'/left_ankle_abnormal')
                left_ankle = im2[int(miny):int(maxy),int(minx):int(maxx)]
                #left_ankle = his(left_ankle)
                cv2.imwrite(os.path.join(sav_dir+'/left_ankle_abnormal', image_name[:-4]+'_left.png'), left_ankle)
                save_coordinate(sav_dir+'/left_ankle_abnormal/coordinate.txt',minx,miny)
                
            elif labels[i].item() is 11:
                if tmp is not 1:
                    if minx.item() < w*0.33:
                        continue
                else:
                    if minx < w*0.33:
                        continue
                if os.path.isdir(sav_dir+'/right_ankle'):
                    l = labels.tolist()
                    if scores[i] > scores[l.index(5)]:
                        shutil.rmtree(sav_dir+'/right_ankle')
                    else:
                        continue
                if os.path.isfile(sav_dir+'/right_ankle_abnormal/coordinate.txt'):
                    f = open(sav_dir+'/right_ankle_abnormal/coordinate.txt')
                    s = f.readline()
                    y = s.split(',')[1]
                    if int(y) < miny.item():
                        os.remove(sav_dir+'/right_ankle_abnormal/coordinate.txt')
                        os.remove(sav_dir+'/right_ankle_abnormal/'+image_name[:-4]+'_right.png')
                    else:
                        continue

                if not os.path.isdir(sav_dir+'/right_ankle_abnormal'):
                     os.makedirs(sav_dir+'/right_ankle_abnormal')
                right_ankle = im2[int(miny):int(maxy),int(minx):int(maxx)]
                #right_ankle = his(right_ankle)
                cv2.imwrite(os.path.join(sav_dir+'/right_ankle_abnormal', image_name[:-4]+'_right.png'), right_ankle)
                save_coordinate(sav_dir+'/right_ankle_abnormal/coordinate.txt',minx,miny)
                
            cv2.drawContours(im2, [box_points], 0, [0,0,255], 20)
            
        cv2.imwrite(os.path.join(sav_dir, image_name), im2)
        print('{} / {} completed!!'.format(cnt,len(image_list)))
        print("Detection time : " + str(time.time() - start_time))
        start_segmentation_engine(image_name[:-4])
        print(time.time() - start_time)
        result_time = time.time() - start_time
        f2 = open(sav_dir+'/time.txt','a')
        f2.write(str(result_time))
        f2.close()

def save_coordinate(path,x,y):
    f1 = open(path,'a')
    f1.write(str(int(x)))
    f1.write(",")
    f1.write(str(int(y)))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch RotatedSSD Training')
    parser.add_argument('--image_size',           type=int,  default=512,                            help='Input image size')
    parser.add_argument('--num_classes',          type=int,  default=13,                              help='Number of classes')

    parser.add_argument('--score_thresh',         type=float, default=0.2,                           help='Gamma update for SGD')
    parser.add_argument('--nms_thresh',           type=float, default=0.4,                           help='Gamma update for SGD')

    parser.add_argument('--load_path',            type=str,  default='./examples/fpnssd/checkpoint/new/ckpt_best.pth',   help='Initial path')
    parser.add_argument('--demo_path',            type=str,  default='./examples/fpnssd/demo/image',                 help='Initial path')
    parser.add_argument('--result_path',          type=str,  default='./examples/fpnssd/demo/result',                help='Initial path')

    args = parser.parse_args()
    
    demo(args)
    

