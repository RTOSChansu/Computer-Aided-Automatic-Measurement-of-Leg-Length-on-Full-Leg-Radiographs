import os
import cv2
import argparse
import random
import numpy as np


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def main(opts):

    root_path = opts.PATH_root
    data_path = os.path.join(root_path, opts.data_dir)

    image_path_train = os.path.join(root_path, 'train')
    image_path_val = os.path.join(root_path, 'val')
    image_path_test = os.path.join(root_path, 'test')

    make_dir(image_path_train)
    make_dir(image_path_val)
    make_dir(image_path_test)

    f_train = open(os.path.join(root_path, 'BA_train.txt'), 'w')
    f_val = open(os.path.join(root_path, 'BA_val.txt'), 'w')
    f_test = open(os.path.join(root_path, 'BA_test.txt'), 'w')

    anno_list = os.listdir(data_path)

    count_total = 0
    count_train = 0
    count_val = 0
    count_test = 0
    for anno_name in anno_list:

        if anno_name.endswith('txt'):

            divider = random.randint(1, 10)
            if divider < 9:
                f = f_train
                im_save_dir = 'train'
                count_train += 1
            elif divider == 10 or divider == 9:
                f = f_val
                im_save_dir = 'val'
                count_val += 1
            '''elif divider == 10:
                f = f_test
                im_save_dir = 'test'
                count_test += 1'''

            with open(os.path.join(data_path, anno_name), 'r') as f_:
                anno_info_list = f_.readlines()

            im = cv2.imread(os.path.join(data_path, anno_name.split('.')[0] + '.jpg'))

            # im_resize = cv2.resize(im, (512, 512))

            im_save_path = os.path.join(root_path, im_save_dir, anno_name.split('.')[0].replace(' ', '_') + '.jpg')
            #im_save_path = os.path.join(root_path, 'test', anno_name.split('.')[0].replace(' ', '_') + '.jpg')
            cv2.imwrite(im_save_path, im)
            f.write(os.path.join(im_save_dir, anno_name.split('.')[0].replace(' ', '_') + '.jpg'))
            print(anno_name)
            height, width, _ = im.shape

            for i, anno_info in enumerate(anno_info_list):

                if i < 1:
                    continue

                anno_info = anno_info.split(',')

                class_num, x, y, w, h, angle = list(map(int, anno_info))
                if angle < 0:
                    angle = 180 + angle
                else:
                    angle = angle

                f.write(' {} {} {} {}'.format(int(x), int(y), int(w), int(h)))
                f.write(' {}'.format(angle))
                f.write(' {}'.format(str(class_num - 1)))
                # f.write(' {}'.format(str(0)))

                # ## For Debug
                # box_points = cv2.boxPoints(((x,y),(w,h),angle))
                # [x1,y1],[x2,y2],[x3,y3],[x4,y4] = box_points
                # x1 = x1/width*512
                # x2 = x2/width*512
                # x3 = x3/width*512
                # x4 = x4/width*512
                # y1 = y1/height*512
                # y2 = y2/height*512
                # y3 = y3/height*512
                # y4 = y4/height*512
                # box_points = np.int0([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
                # cv2.drawContours(im_resize, [box_points], 0, (255,0,0), 2)
                # cv2.putText(im_resize, text='{0:.4f}'.format(angle), org=(int(x1), int(y2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255,0,0))


            f.write('\n')
            count_total += 1
            print('[Total: {}][Train: {}][Val: {}][Test: {}] Preprocessed!'.format(count_total, count_train, count_val, count_test))

            #print(anno_name)
            # cv2.imshow('image', im_resize)
            # cv2.waitKey()

    f_train.close()
    f_val.close()
    f_test.close()

if __name__ == '__main__':
    opts = argparse.ArgumentParser()
    opts.add_argument('--PATH_root', default='data/ba')
    opts.add_argument('--data_dir', default='data_final')

    opts = opts.parse_args()

    main(opts)
