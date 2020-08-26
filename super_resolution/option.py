import argparse
from . import utility
import numpy as np
from easydict import EasyDict as edict


args = edict()
args.n_threads = 4
args.cpu = False
args.n_GPUs = 1
args.seed = 1
args.data_train = 'AppleCT'
args.data_test = 'AppleCT'
args.scale = 2
args.patch_size = 48
args.rgb_range = 1
args.n_colors = 1
args.no_augment = True
args.model = 'DRN-L'
args.pre_train = '.'
args.pre_train_dual = '.'
args.n_blocks = 30
args.n_feats = 16
args.negval = 0.2
args.test_every = 1000
args.epochs = 100
args.batch_size = 32
args.self_ensemble = False
args.test_only = True
args.lr = 1e-4
args.eta_min = 1e-7
args.beta1 = 0.9
args.beta2 = 0.999
args.epsilon = 1e-8
args.weight_decay = 0
args.loss = '1*L1'
args.skip_threshold = 1e6
args.dual_weight = 0.1
args.save = './experiment/test/'
args.print_every = 100
args.save_results = False


utility.init_model(args)

# scale = [2,4] for 4x SR to load data
# scale = [2,4,8] for 8x SR to load data
args.scale = [pow(2, s+1) for s in range(int(np.log2(args.scale)))]

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

