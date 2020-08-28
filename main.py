import os
import torch
import argparse
from tqdm import tqdm

from denoise.predict import DenoiseModel
from super_resolution.predict import SRModel
from wgan_vgg.predict import RecModel
from fbp import FBP
from tools import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help='input data dir')
    parser.add_argument('--save_dir', type=str, default='tmp', help='save the test result')
    parser.add_argument('--data_type', type=str, required=True,
            choices=['noisefree', 'gaussian', 'scattering'], help='input data type')
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
    inp_list = get_file_list(args.data_dir, suffix=['tif', 'npy'])
    print('total input data: {}'.format(len(inp_list)))
    sr_model = SRModel(args.data_type, args.device)
    rec_model = RecModel(args.device)
    if args.data_type != 'noisefree':
        denoise_model = DenoiseModel(args.data_type, args.device)
    print('models loaded!')
    fbp = FBP()

    for inp_path in tqdm(inp_list):
        inp_data = get_data(inp_path, norm=1.0)
        if inp_data.shape[0] == 50:
            # 25 projections
            inp_data = inp_data[::2]
        inp_data = torch.from_numpy(inp_data)
        inp_data = inp_data.unsqueeze(0).unsqueeze(0).to(args.device)
        if args.data_type != 'noisefree':
            inp_data = denoise_model.pred(inp_data)
        sr_data = sr_model.pred(inp_data)
        sr_data = sr_data.squeeze(0).squeeze(0).cpu().numpy()
        fbp_data = fbp.fbp_res(sr_data)
        fbp_data = torch.from_numpy(fbp_data).float()
        fbp_data = fbp_data.unsqueeze(0).unsqueeze(0).to(args.device)
        rec_data = rec_model.pred(fbp_data)
        rec_data = rec_data.numpy()
        data_name = inp_path.split('/')[-1].split('.')[0]
        save_data(args.save_dir, data_name, rec_data)


if __name__ == '__main__':
    main()



