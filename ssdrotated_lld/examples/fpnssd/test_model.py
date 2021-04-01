import os
#import sys
#sys.path.insert(0, '/home/sonic/lld/ssdrotated_lld')
import time

from test_l_ankle import segmentation_l_ankle
from test_r_ankle import segmentation_r_ankle
from test_l_knee import segmentation_l_knee
from test_r_knee import segmentation_r_knee
from test_l_pelvis import segmentation_l_pelvis
from test_r_pelvis import segmentation_r_pelvis

from test_l_ankle_ab import segmentation_l_ankle_ab
from test_r_ankle_ab import segmentation_r_ankle_ab
from test_l_knee_ab import segmentation_l_knee_ab
from test_r_knee_ab import segmentation_r_knee_ab
from test_l_pelvis_ab import segmentation_l_pelvis_ab
from test_r_pelvis_ab import segmentation_r_pelvis_ab

from mapping import point_mapping

path = '/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/'

def start_segmentation_engine(name):
    start = time.time()
    if os.path.isdir(path+name+'/left_pelvis'):
        segmentation_l_pelvis(name)
    elif os.path.isdir(path+name+'/left_pelvis_abnormal'):
        segmentation_l_pelvis_ab(name)
    
    if os.path.isdir(path+name+'/right_pelvis'):
        segmentation_r_pelvis(name)
    elif os.path.isdir(path+name+'/right_pelvis_abnormal'):
        segmentation_r_pelvis_ab(name)
        
    if os.path.isdir(path+name+'/left_knee'):
        segmentation_l_knee(name)
    elif os.path.isdir(path+name+'/left_knee_abnormal'):
        segmentation_l_knee_ab(name)
        
    if os.path.isdir(path+name+'/right_knee'):
        segmentation_r_knee(name)
    elif os.path.isdir(path+name+'/right_knee_abnormal'):
        segmentation_r_knee_ab(name)
        
    if os.path.isdir(path+name+'/left_ankle'):
        segmentation_l_ankle(name)
    elif os.path.isdir(path+name+'/left_ankle_abnormal'):
        segmentation_l_ankle_ab(name)
        
    if os.path.isdir(path+name+'/right_ankle'):
        segmentation_r_ankle(name)
    elif os.path.isdir(path+name+'/right_ankle_abnormal'):
        segmentation_r_ankle_ab(name)
    print("Segmentation time: "+ str(time.time()-start))
    print("mappping")
    point_mapping(name)
    print("finish")
    
