
import torch
import argparse
import numpy as np

from models import BRDNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help="train | test")
    parser.add_argument('--pretrained', type=str, default='./ckpt/b8xn10xs50/model_checkpoint_35.pth', help="pretrained model")
    parser.add_argument('--train_path', type=str, default='../data/prj_denoise/train.txt')
    parser.add_argument('--val_path', type=str, default='../data/prj_denoise/val.31101.txt')
    parser.add_argument('--save_dir', type=str, default='./ckpt/1')

    # if patch training, batch size is (--patch_n x --batch_size)
    parser.add_argument('--patch_n', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)

    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=3000)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--test_interval', type=int, default=1)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    return args


class DenoiseModel(object):

    def __init__(self):
        args = get_args()
        self.device = args.device
        self.model = BRDNet()
        self.model.load_state_dict(torch.load(args.pretrained))
        self.model.to(self.device)
        self.model.eval()

    def pred(inp_data):
        pred = model(inp_data)
        pred[pred < 0] = 0
        pred[pred > 1] = 1
        return pred


