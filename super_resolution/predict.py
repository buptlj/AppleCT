import torch
from .model import Model
from .option import args
from .checkpoint import Checkpoint
from .utility import set_seed


class SRModel(object):

    def __init__(self, device):
        #args.pre_train = model_path
        args.pre_train = 'super_resolution/ckpt/model_best.pth'
        args.device_id = device
        set_seed(args.seed)
        checkpoint = Checkpoint(args)
        self.drn_model = Model(args, checkpoint)
        self.drn_model.eval()

    def pred(self, inp_data):
        with torch.no_grad():
            sr = self.drn_model(inp_data)
            if isinstance(sr, list): sr = sr[-1]
            sr = sr.clamp(0, 1)
        return sr


