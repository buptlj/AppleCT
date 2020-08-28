
import torch
import argparse
import numpy as np

from .models import BRDNet


class DenoiseModel(object):

    def __init__(self, data_type, device):
        self.data_type = data_type
        self.device = device
        self.model = BRDNet()
        if data_type == 'gaussian':
            model_path = 'denoise/ckpt/gaussian_model_best.pth'
        elif data_type == 'scattering':
            model_path = 'denoise/ckpt/scattering_model_best.pth'
        else:
            model_path = ''
            print('wrong data type!')
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def pred(self, inp_data):
        with torch.no_grad():
            pred = self.model(inp_data)
            pred[pred < 0] = 0
            pred[pred > 1] = 1
        return pred


