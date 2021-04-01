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

    anno_list = os.listdir(data_path)

    count_total = 0
    count_train = 0
    count_val = 0
    count_test = 0
    for anno_name in anno_list:

        if anno_name.endswith('txt'):
            divider = random.randint(1, 10)
            if divider < 9:
                count_train += 1
            elif divider == 9:
                count_val += 1
            else:
                count_test += 1

            with open(os.path.join(data_path, anno_name), 'r') as f_:
                anno_info_list = f_.readlines()

            im = cv2.imread(os.path.join(data_path, anno_name.split('.')[0] + '.jpg'))

            im_resize = cv2.resize(im, (512, 512))

            height, width, _ = im.shape

            for i, anno_info in enumerate(anno_info_list):

                if i < 1:
                    continue

                anno_info = anno_info.split(',')

                class_num, x, y, w, h, angle = list(map(int, anno_info))
                if angle < 0:
                    angle_temp = 180 + angle
                else:
                    angle_temp = angle

                # ## by sj
                # if height <= width:
                #     if angle_temp == -0:
                #         angle = 0
                #     elif angle_temp == -90:
                #         angle = 90
                #     else:
                #         angle = -angle
                # else:
                #     if angle_temp == -0:
                #         angle = 90
                #     elif angle_temp == -90:
                #         angle = 0
                #     else:
                #         angle = 90 - angle
                #
                # if height <= width:
                #     if angle_temp == -0:
                #         angle = 0
                #     elif angle_temp == 90:
                #         angle = -90
                #     else:
                #         angle = -angle
                # else:
                #     if angle_temp == 90:
                #         angle = 0
                #     elif angle_temp == 0:
                #         angle = -90
                #     else:
                #         angle = 90 - angle

                ## For Debug
                box_points = cv2.boxPoints(((x,y),(w,h),angle_temp))
                [x1,y1],[x2,y2],[x3,y3],[x4,y4] = box_points
                x = x/width*512
                y = y/height*512
                x1 = x1/width*512
                x2 = x2/width*512
                x3 = x3/width*512
                x4 = x4/width*512
                y1 = y1/height*512
                y2 = y2/height*512
                y3 = y3/height*512
                y4 = y4/height*512
                box_points = np.int0([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
                cv2.drawContours(im_resize, [box_points], 0, (255,0,0), 2)
                cv2.putText(im_resize, text='{}'.format(class_num), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255,0,0))
                # cv2.putText(im_resize, text='{0:.4f}'.format(angle_temp), org=(int(x1), int(y2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255,0,0))


            count_total += 1
            print('[Total: {}][Train: {}][Val: {}][Test: {}] Preprocessed!'.format(count_total, count_train, count_val, count_test))

            print(anno_name)
            cv2.imshow('image', im_resize)
            cv2.waitKey()

if __name__ == '__main__':
    opts = argparse.ArgumentParser()
    opts.add_argument('--PATH_root', default='data/ba')
    opts.add_argument('--data_dir', default='data_final')

    opts = opts.parse_args()

    main(opts)
