import os
import cv2
import argparse

def main(opts):
    root_path = opts.PATH_root
    data_path = os.path.join(root_path, opts.data_dir)
    origin_path = os.path.join(root_path, 'data_final')
    save_path = os.path.join(root_path, opts.save_dir)
    anno_list = os.listdir(data_path)
    path = '/home/sonic/lld/ssdrotated_lld/'
    print('start...')

    for anno_name in anno_list:
        coordinate = []
        with open(os.path.join(origin_path, anno_name.split('.')[0]+'.txt'), 'r') as f_:
            anno_info_list = f_.readlines()
            s = 'class, center_x, center_y, w, h, angle'
            for i, anno_info in enumerate(anno_info_list):
                if i < 1:
                    continue
                anno_info = anno_info.split(',')
                class_num, x, y, w, h, angle = list(map(int, anno_info))
                coordinate.append([class_num, x, y, w, h, angle])
            save = open(path+save_path+'/'+anno_name.split('.')[0]+'.txt','w')
            save.write(s+'\n')
            for i in range(0,len(coordinate)):
                 if int(coordinate[i][0]) is 7:
                     coordinate[i][0] = 1
                 elif int(coordinate[i][0]) is 8:
                     coordinate[i][0] = 2
                 save.write('{},{},{},{},{},{}'.format(int(coordinate[i][0]),int(coordinate[i][1]),int(coordinate[i][2]),int(coordinate[i][3]),int(coordinate[i][4]),int(coordinate[i][5])))
                 save.write('\n')
            save.close()

if __name__ == '__main__':
    opts = argparse.ArgumentParser()
    opts.add_argument('--PATH_root', default='data/ba')
    opts.add_argument('--data_dir', default='changed_imgs')
    opts.add_argument('--save_dir', default='changed_labels')
    opts = opts.parse_args()
    main(opts)
