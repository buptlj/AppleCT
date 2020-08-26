
import torch
import argparse
import numpy as np

from .models import BRDNet


class DenoiseModel(object):

    def __init__(self, data_type, device):
        self.data_type = data_type
        self.device = device
        self.model = BRDNet()
        model_path = 'denoise/ckpt/model_best.pth'
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def pred(self, inp_data):
        with torch.no_grad():
            pred = self.model(inp_data)
            pred[pred < 0] = 0
            pred[pred > 1] = 1
        return pred


