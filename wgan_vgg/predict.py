import os
import torch
import argparse

from networks import WGAN_VGG, WGAN_VGG_generator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--model_path', type=str, default='./save/')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--norm_range_min', type=float, default=0.0)
    parser.add_argument('--norm_range_max', type=float, default=255.0)
    parser.add_argument('--trunc_min', type=float, default=0.0)
    parser.add_argument('--trunc_max', type=float, default=255.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    print(args)
    return args

class RecModel(object):

    def __init__(self):
        args = get_args()
        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.model_path = args.model_path
        self.model = WGAN_VGG(device=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def pred(self, inp_data):
        # inp_data: the result of fbp, range 0-1
        shape_ = inp_data.shape[-1]
        with torch.no_grad():
            pred = self.model.generator(inp_data)
            pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))
        return pred


