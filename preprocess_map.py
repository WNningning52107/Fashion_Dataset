import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from utils.util import get_map


def main(opt):
    name_list = os.listdir(opt.sketch_path)
    target = name_list
    for name in tqdm(target):
        if os.path.exists(os.path.join(opt.save_path, os.path.splitext(name)[0]+".npy")): 
            continue
        else:
            x = cv2.imread(os.path.join(opt.sketch_path, name), cv2.IMREAD_GRAYSCALE)
            _, line = cv2.threshold(x, 220, 255, cv2.THRESH_BINARY)
            line_map = get_map(line)
            np.save(os.path.join(opt.save_path, os.path.splitext(name)[0]), line_map)
            print('save map to', os.path.join(opt.save_path, os.path.splitext(name)[0]))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='')
    args.add_argument('--sketch_path', default='/home/wn/Fashion_Dataset/train/line_art/line_art_binary')
    args.add_argument('--save_path', default='/home/wn/Fashion_Dataset/train/map')
    args = args.parse_args()

    #os.makedirs(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)

    main(args)