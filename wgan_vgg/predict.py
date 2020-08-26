import os
import torch

from .networks import WGAN_VGG, WGAN_VGG_generator



class RecModel(object):

    def __init__(self, device):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = 0.0
        self.norm_range_max = 255.0
        self.trunc_min = 0.0
        self.trunc_max = 255.0

        self.model_path = 'wgan_vgg/ckpt/WGANVGG_52000iter.ckpt'
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


