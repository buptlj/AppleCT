import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import argparse
from tqdm import tqdm

from .unet import UNet


def preprocess(pil_img, scale):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    pil_img = pil_img.resize((newW, newH))

    img_nd = np.array(pil_img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2).repeat(3, axis=2)

    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    return img_trans


class DefectSeg(object):
    def __init__(self, weights="defect_det/ckpt/model.pth", device="cuda:0"):
        self.net = UNet(n_channels=3, n_classes=5).to(device=device)
        self.net.load_state_dict(torch.load(weights, map_location=device))
        self.device = device
        self.net.eval()

    def seg(self, img_path):
        img = Image.open(img_path)
        img = torch.from_numpy(preprocess(img, scale=1))

        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.net(img)

            if self.net.n_classes > 1:
                probs = F.softmax(output, dim=1)
            else:
                probs = torch.sigmoid(output)

            probs = probs.squeeze(0)

            probs = probs.argmax(dim=0)

            probs[probs==1] = 64
            probs[probs==2] = 128
            probs[probs==3] = 191
            probs[probs==2] = 255

        return probs.cpu().numpy()



if __name__ == "__main__":
    weights = "ckpt/epoch_2.pth"
    device = "cuda:0"
    apple_seg = DefectSeg(weights=weights, device=device)

    img_path = "31102_200.png"
    mask = apple_seg.seg(img_path)
    cv2.imwrite("mask.jpg", mask)
