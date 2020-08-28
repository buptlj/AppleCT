import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import argparse
from tqdm import tqdm

from defect_det.predict import DefectSeg
from tools import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help='input data dir')
    parser.add_argument('--save_dir', type=str, default='tmp', help='save the test result')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    if not os.path.exists(args.data_dir):
        print('input data dir not exists!')
        return
    if not os.path.exists(args.save_dir):
        print('save dir not exists!, make dir ', args.save_dir)
        os.makedirs(args.save_dir)
    inp_list = get_file_list(args.data_dir, suffix=['tif', 'jpg', 'png'])
    print('total input data: {}'.format(len(inp_list)))

    apple_seg = DefectSeg(device=args.device)
    print('model loaded!')
    for inp_path in tqdm(inp_list):
        seg_res = apple_seg.seg(inp_path)
        data_name = inp_path.split('/')[-1].split('.')[0]
        dst_path = os.path.join(args.save_dir, data_name + '.jpg')
        cv2.imwrite(dst_path, seg_res)



if __name__ == "__main__":
    main()


