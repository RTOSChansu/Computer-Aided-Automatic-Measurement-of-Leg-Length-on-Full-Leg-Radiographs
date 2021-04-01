import os
import cv2
import argparse

def main(opts):
    root_path = opts.PATH_root
    data_path = os.path.join(root_path, opts.data_dir)
    save_path = os.path.join(root_path, opts.save_dir)
    anno_list = os.listdir(data_path)
    path = '/home/sonic/lld/ssdrotated_lld/'
    print('start...')
    crop_thres = 5

    for anno_name in anno_list:
        coordinate = []
        if anno_name.endswith('txt'):
            im = cv2.imread(os.path.join(data_path, anno_name.split('.')[0] + '.jpg'))
            #im2 = cv2.flip(im,1)
            #if os.path.isfile(path+save_path+'/'+anno_name.split('.')[0]+'_1.jpg'):
            #    continue
            #else:
            #    cv2.imwrite(path+save_path+'/'+anno_name.split('.')[0]+'_1.jpg',im2)

            #print(anno_name)

            with open(os.path.join(data_path, anno_name), 'r') as f_:
                anno_info_list = f_.readlines()
            coordinate = []
            s = 'class, center_x, center_y, w, h, angle'
            for i, anno_info in enumerate(anno_info_list):
                if i < 1:
                    continue
                anno_info = anno_info.split(',')
                
                class_num, x, y, w, h, angle = list(map(int, anno_info))
                coordinate.append([class_num, x, y, w, h, angle])
            #if os.path.isfile(path+save_path+'/'+anno_name.split('.')[0]+'_1.txt'):
            #    continue
            #else:
            im_save_path = os.path.join(save_path,anno_name.split('.')[0])
            #save = open(path+save_path+'/'+anno_name.split('.')[0]+'_1.txt','w')
            #save.write(s+'\n')
            rows, cols = im.shape[:2]
            print(anno_name)

            #crop
            #left top imgname_1
            print(coordinate)
            im_1 = im[0:rows - crop_thres, 0:cols - crop_thres]
            cv2.imwrite(im_save_path+'_1.jpg',im_1)
            f = open(im_save_path+'_1.txt','w')
            f.write(s+"\n")
            for i in range(0,6):
                f.write('{},{},{},{},{},{}'.format(int(coordinate[i][0]),int(coordinate[i][1]),int(coordinate[i][2]-crop_thres),int(coordinate[i][3]),int(coordinate[i][4]),int(coordinate[i][5])))
                f.write('\n')
            f.close()

            #left bottom imgname_2
            im_2 = im[crop_thres:rows, 0:cols - crop_thres]
            cv2.imwrite(im_save_path+'_2.jpg',im_2)
            f = open(im_save_path+'_2.txt','w')
            f.write(s+"\n")
            for i in range(0,6):
                f.write('{},{},{},{},{},{}'.format(int(coordinate[i][0]),int(coordinate[i][1]),int(coordinate[i][2]-crop_thres),int(coordinate[i][3]),int(coordinate[i][4]),int(coordinate[i][5])))
                f.write('\n')
            f.close()

            #right top imgname_3
            im_3 = im[0:rows - crop_thres, crop_thres:cols]
            cv2.imwrite(im_save_path+'_3.jpg',im_3)
            f = open(im_save_path+'_3.txt','w')
            f.write(s+"\n")
            for i in range(0,6):
                f.write('{},{},{},{},{},{}'.format(int(coordinate[i][0]),int(coordinate[i][1]-crop_thres),int(coordinate[i][2]-crop_thres),int(coordinate[i][3]),int(coordinate[i][4]),int(coordinate[i][5])))
                f.write('\n')
            f.close()

            #right bottom imgname_4
            im_4 = im[crop_thres:rows, crop_thres:cols]
            cv2.imwrite(im_save_path+'_4.jpg',im_4)
            f = open(im_save_path+'_4.txt','w')
            f.write(s+"\n")
            for i in range(0,6):
                f.write('{},{},{},{},{},{}'.format(int(coordinate[i][0]),int(coordinate[i][1]-crop_thres),int(coordinate[i][2]-crop_thres),int(coordinate[i][3]),int(coordinate[i][4]),int(coordinate[i][5])))
                f.write('\n')
            f.close()

            #middle imgname_5
            im_5 = im[crop_thres:rows - crop_thres, crop_thres:cols - crop_thres]
            cv2.imwrite(im_save_path+'_5.jpg',im_5)
            f = open(im_save_path+'_5.txt','w')
            f.write(s+"\n")
            for i in range(0,6):
                f.write('{},{},{},{},{},{}'.format(int(coordinate[i][0]),int(coordinate[i][1]-crop_thres),int(coordinate[i][2]-2*crop_thres),int(coordinate[i][3]),int(coordinate[i][4]),int(coordinate[i][5])))
                f.write('\n')
            f.close()

if __name__ == '__main__':
    opts = argparse.ArgumentParser()
    opts.add_argument('--PATH_root', default='data/ba')
    opts.add_argument('--data_dir', default='abnormal')
    opts.add_argument('--save_dir', default='aug')
    opts = opts.parse_args()
    main(opts)
