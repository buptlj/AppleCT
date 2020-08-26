import numpy as np 
from scipy.fftpack import fft, fftshift, ifft
from PIL import Image

from tools import norm_data


class FBP(object):

    def __init__(self, thetas=np.linspace(88.22, -91.78, 50, endpoint=False)):
        self.thetas = thetas

    def filter_projection(self, sinogram):  ##ramp filter
        a = 0.1
        num_thetas, size = sinogram.shape
        step = 2 * np.pi / size
        w = np.arange(-np.pi, np.pi, step)
        if len(w) < size:
            w = np.concatenate([w, w[-1] + step])
        rn1 = np.abs(2 / a * np.sin(a * w / 2))
        rn2 = np.sin(a * w / 2) / (a * w / 2)
        r = rn1 * (rn2) ** 2

        filter_ = fftshift(r)

        filter_sinogram = np.zeros((num_thetas, size))
        for i in range(num_thetas):
            proj_fft = fft(sinogram[i])
            filter_proj = proj_fft * filter_
            filter_sinogram[i] = np.real(ifft(filter_proj))
        return filter_sinogram

    def back_projection(self, sinogram):
        size_ = sinogram.shape[1]
        recon_img = np.zeros((size_, size_))

        for i, theta in enumerate(self.thetas):
            tmp1 = sinogram[i, :]

            tmp = np.repeat(np.expand_dims(tmp1, 1), size_, axis=1).T
            tmp = Image.fromarray(tmp)

            tmp = tmp.rotate(theta, expand=0)
            #tmp = tmp.transpose(Image.FLIP_TOP_BOTTOM)
            recon_img += tmp

        return np.flipud(recon_img)

    def fbp_res(self, prj_img):
        ih = prj_img.shape[0]
        self.thetas = np.linspace(88.22, -91.78, ih, endpoint=False)
        filted = self.filter_projection(prj_img)
        # Back project the convolved sinogram 
        reconstruction = self.back_projection(filted)
        reconstruction = reconstruction[202:1174, 202:1174]
        reconstruction = norm_data(reconstruction)
        return reconstruction

