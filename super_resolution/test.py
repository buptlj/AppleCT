import torch
import model
from option import args
from checkpoint import Checkpoint
import utility


class SRModel(object):

    def __init__(self):
        #args.pre_train = model_path
        utility.set_seed(args.seed)
        checkpoint = Checkpoint(args)
        self.drn_model = model.Model(args, checkpoint)
        self.drn_model.eval()

    def pred(self, inp_data):
        with torch.no_grad():
            sr = self.drn_model(inp_data)
            if isinstance(sr, list): sr = sr[-1]
            sr = sr.clamp(0, 1)
        return sr


