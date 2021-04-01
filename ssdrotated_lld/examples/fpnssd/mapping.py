import os
import cv2
import math

root = "/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/image"
save_path = "/home/sonic/lld/ssdrotated_lld/examples/fpnssd/demo/result/"

def point_mapping(n):

    ## left leg length
    im_list = os.listdir(root)
    im = cv2.imread(os.path.join(root,n+'.jpg'))
    
    im2 = im.copy()
    if os.path.isdir(save_path+ n +"/left_pelvis"):
        f = open(save_path+ n +"/left_pelvis/coordinate_point.txt",'r')
    elif os.path.isdir(save_path+ n +"/left_pelvis_abnormal"):
        f = open(save_path+ n +"/left_pelvis_abnormal/coordinate_point.txt",'r')
    point = f.readlines()
    
    pelvis_topx = point[0].split(',')[0]
    pelvis_topy = point[0].split(',')[1]
    #pelvis_cx = point[1].split(',')[0]
    #pelvis_cy = point[1].split(',')[1]
    #pelvis_stx = point[2].split(',')[0]
    #pelvis_sty = point[2].split(',')[1]

    f.close()
    
    if os.path.isdir(save_path+ n +"/left_ankle"):
        f = open(save_path+ n +"/left_ankle/coordinate_point.txt","r")
    elif os.path.isdir(save_path+ n +"/left_ankle_abnormal"):
        f = open(save_path+ n +"/left_ankle_abnormal/coordinate_point.txt","r")
    point = f.readline()
    
    ankle_cenx = point.split(',')[0]
    ankle_ceny = point.split(',')[1]
    
    f.close()
    
    im_1 = cv2.line(im,(int(pelvis_topx),int(pelvis_topy)),(int(ankle_cenx),int(ankle_ceny)),(0,255,0),20)
    left_leg_length = math.sqrt(pow(int(pelvis_topx)-int(ankle_cenx),2) + pow(int(ankle_ceny)-int(pelvis_topy),2))
    
    ## left femoral length
    if os.path.isdir(save_path+ n +"/left_knee"):
        f = open(save_path+ n +"/left_knee/coordinate_point.txt",'r')
    elif os.path.isdir(save_path+ n +"/left_knee_abnormal"):
        f = open(save_path+ n +"/left_knee_abnormal/coordinate_point.txt",'r')
    point = f.readlines()
    
    knee_rx = point[0].split(',')[0]
    knee_ry = point[0].split(',')[1]
    
    #knee_cx = point[1].split(',')[0]
    #knee_cy = point[1].split(',')[1]
    f.close()
    
    im_2 = cv2.line(im,(int(pelvis_topx),int(pelvis_topy)),(int(knee_rx),int(knee_ry)),(255,0,0),20)
    left_femoral_length = math.sqrt(pow(int(pelvis_topx)-int(knee_rx),2) + pow(int(knee_ry)-int(pelvis_topy),2))
    
    ## left tibial length
    im_3 = cv2.line(im,(int(knee_rx),int(knee_ry)),(int(ankle_cenx),int(ankle_ceny)),(0,0,255),20)
    left_tibial_length = math.sqrt(pow(int(ankle_cenx)-int(knee_rx),2) + pow(int(knee_ry)-int(ankle_ceny),2))
    
    ## print text
    h, w, c = im_3.shape
    cv2.putText(im_3,'RLL : ' + str(round(int(left_leg_length*0.1395)*0.1,2))+'cm', org=(int(w/2),int(pelvis_topy)),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=6.5, color=(0, 0, 255), thickness=10)
    cv2.putText(im_3,'RFL : ' + str(round(int(left_femoral_length*0.1395)*0.1,2))+'cm', org=(int(w/2),int(pelvis_topy)+300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=6.5, color=(0, 0, 255), thickness=10)
    cv2.putText(im_3,'RTL : ' + str(round(int(left_tibial_length*0.1395)*0.1,2))+'cm', org=(int(w/2),int(pelvis_topy)+600), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=6.5, color=(0, 0, 255), thickness=10)
    '''## Left proximal femoral line
    im_3_1 = cv2.line(im,(int(pelvis_cx),int(pelvis_cy)),(int(pelvis_stx),int(pelvis_sty)),(0,0,255),20)
    left_proximal_femoral_line = math.sqrt(pow(int(pelvis_cx)-int(pelvis_cy),2) + pow(int(pelvis_stx)-int(pelvis_sty),2))'''

    ## Calculate HKA
    #im_4 = cv2.line(im,(int(pelvis_cx),int(pelvis_cy)),(int(knee_cx),int(knee_cy)),(0,255,255),20)
    #im_5 = cv2.line(im,(int(knee_cx),int(knee_cy)),(int(ankle_cenx),int(ankle_ceny)),(0,255,255),20)
    
    '''a = math.sqrt(pow(int(pelvis_cx) - int(ankle_cenx),2) + pow(int(pelvis_cy) - int(ankle_ceny),2))
    b = math.sqrt(pow(int(pelvis_cx) - int(knee_cx),2) + pow(int(pelvis_cy) - int(knee_cy),2))
    c = math.sqrt(pow(int(knee_cx) - int(ankle_cenx),2) + pow(int(knee_cy) - int(ankle_ceny),2))
    
    tmp = (pow(b,2) + pow(c,2) - pow(a,2)) / (2*b*c) 
    
    angle = math.acos(tmp)
    if angle * (180/math.pi) < 180 :
        angle = 180 - angle * (180/math.pi)
    else:
        angle = 180 + angle * (180/math.pi)'''
        
    f = open(save_path+ n +"/information.txt",'a')
    f.write(str(int(left_leg_length*0.1395)))
    f.write("\n")
    f.write(str(int(left_femoral_length*0.1395)))
    f.write("\n")
    f.write(str(int(left_tibial_length*0.1395)))
    f.write("\n")
    #f.write(str(int(angle)))
    #f.write("\n")
    #f.write(str(int(left_proximal_femoral_line)))
    f.close()

    cv2.imwrite(save_path+n+"/"+n+'_left_leg.jpg',im_3)
    
    ## rigth leg length
    if os.path.isdir(save_path+ n +"/right_pelvis"):
        f = open(save_path+ n +"/right_pelvis/coordinate_point.txt",'r')
    elif os.path.isdir(save_path+ n +"/right_pelvis_abnormal"):
        f = open(save_path+ n +"/right_pelvis_abnormal/coordinate_point.txt",'r')
    point = f.readlines()
    
    pelvis_topxx = point[0].split(',')[0]
    pelvis_topyy = point[0].split(',')[1]
    #pelvis_cxx = point[1].split(',')[0]
    #pelvis_cyy = point[1].split(',')[1]
    #pelvis_stxx = point[2].split(',')[0]
    #pelvis_styy = point[2].split(',')[1]
    f.close()
    
    if os.path.isdir(save_path+ n +"/right_ankle"):
        f = open(save_path+ n +"/right_ankle/coordinate_point.txt","r")
    elif os.path.isdir(save_path+ n +"/right_ankle_abnormal"):
        f = open(save_path+ n +"/right_ankle_abnormal/coordinate_point.txt","r")
    point = f.readline()
    
    ankle_cenxx = point.split(',')[0]
    ankle_cenyy = point.split(',')[1]
    
    f.close()
    
    im_6 = cv2.line(im2,(int(pelvis_topxx),int(pelvis_topyy)),(int(ankle_cenxx),int(ankle_cenyy)),(0,255,0),20)
    right_leg_length = math.sqrt(pow(int(pelvis_topxx)-int(ankle_cenxx),2) + pow(int(pelvis_topyy)-int(ankle_cenyy),2))
    
    ## right femoral length
    if os.path.isdir(save_path+ n +"/right_knee"):
        f = open(save_path+ n +"/right_knee/coordinate_point.txt",'r')
    elif os.path.isdir(save_path+ n +"/right_knee_abnormal"):
        f = open(save_path+ n +"/right_knee_abnormal/coordinate_point.txt",'r')
    point = f.readlines()
    
    knee_lx = point[0].split(',')[0]
    knee_ly = point[0].split(',')[1]
    #knee_cxx = point[1].split(',')[0]
    #knee_cyy = point[1].split(',')[1]
    
    f.close()
    
    im_7 = cv2.line(im2,(int(pelvis_topxx),int(pelvis_topyy)),(int(knee_lx),int(knee_ly)),(255,0,0),20)
    right_femoral_length = math.sqrt(pow(int(pelvis_topxx)-int(knee_lx),2) + pow(int(pelvis_topyy)-int(knee_ly),2))
    
    ## right tibial length
    im_8 = cv2.line(im2,(int(knee_lx),int(knee_ly)),(int(ankle_cenxx),int(ankle_cenyy)),(0,0,255),20)
    right_tibial_length = math.sqrt(pow(int(ankle_cenxx)-int(knee_lx),2) + pow(int(ankle_cenyy)-int(knee_ly),2))
    
    ## print text
    cv2.putText(im_8,'LLL : ' + str(round(int(right_leg_length*0.1395)*0.1,2))+'cm', org=(10,int(pelvis_topy)),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=6.5, color=(0, 255, 255), thickness=10)
    cv2.putText(im_8,'LFL : ' + str(round(int(right_femoral_length*0.1395)*0.1,2))+'cm', org=(10,int(pelvis_topy)+300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=6.5, color=(0, 255, 255), thickness=10)
    cv2.putText(im_8,'LTL : ' + str(round(int(right_tibial_length*0.1395)*0.1,2))+'cm', org=(10,int(pelvis_topy)+600), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=6.5, color=(0, 255, 255), thickness=10)
    
    '''## right proximal femoral line
    im_8_1 = cv2.line(im2,(int(pelvis_cxx),int(pelvis_cyy)),(int(pelvis_stxx),int(pelvis_styy)),(0,0,255),20)
    right_proximal_femoral_line = math.sqrt(pow(int(pelvis_cxx)-int(pelvis_cyy),2) + pow(int(pelvis_stxx)-int(pelvis_styy),2))'''
    
    ## Calculate HKA
    #im_9 = cv2.line(im2,(int(pelvis_cxx),int(pelvis_cyy)),(int(knee_cxx),int(knee_cyy)),(0,255,255),20)
    #im_10 = cv2.line(im2,(int(knee_cxx),int(knee_cyy)),(int(ankle_cenxx),int(ankle_cenyy)),(0,255,255),20)
    
    '''a = math.sqrt(pow(int(pelvis_cxx) - int(ankle_cenxx),2) + pow(int(pelvis_cyy) - int(ankle_cenyy),2))
    b = math.sqrt(pow(int(pelvis_cxx) - int(knee_cxx),2) + pow(int(pelvis_cyy) - int(knee_cyy),2))
    c = math.sqrt(pow(int(knee_cxx) - int(ankle_cenxx),2) + pow(int(knee_cyy) - int(ankle_cenyy),2))
    
    tmp = (pow(b,2) + pow(c,2) - pow(a,2)) / (2*b*c) 
    
    angle = math.acos(tmp)
    if angle * (180/math.pi) < 180 :
        angle = 180 - angle * (180/math.pi)
    else:
        angle = 180 + angle * (180/math.pi)'''
    
    
    f = open(save_path+ n +"/information.txt",'a')
    f.write(str(int(right_leg_length*0.1395)))
    f.write("\n")
    f.write(str(int(right_femoral_length*0.1395)))
    f.write("\n")
    f.write(str(int(right_tibial_length*0.1395)))
    #f.write("\n")
    #f.write(str(int(angle)))
    #f.write("\n")
    #f.write(str(int(right_proximal_femoral_line)))
    f.close()
    
    cv2.imwrite(save_path+n+"/"+n+'_right_leg.jpg',im_8)

        
if __name__ == '__main__':
    point_mapping(n)
