from math import floor, ceil

import numpy as np
import torch
import torch.nn as nn

from cross_correlation import xcorr_torch as ccorr
from spectral_tools import gen_mtf
from utils import local_corr_mask

class SpectralLossNocorr(nn.Module):
    def __init__(self, mtf, net_crop, pan_shape, ratio, device, mask=None):

        # Class initialization
        super(SpectralLossNocorr, self).__init__()
        kernel = mtf
        # Parameters definition
        self.nbands = kernel.shape[-1]
        self.net_scope = net_crop
        self.device = device
        self.ratio = ratio

        # Conversion of filters in Tensor
        self.MTF_r = 2
        self.MTF_c = 2
        self.pad = floor((kernel.shape[0] - 1) / 2)

        kernel = np.moveaxis(kernel, -1, 0)
        kernel = np.expand_dims(kernel, axis=1)

        kernel = torch.from_numpy(kernel).type(torch.float32)

        # DepthWise-Conv2d definition
        self.depthconv = nn.Conv2d(in_channels=self.nbands,
                                   out_channels=self.nbands,
                                   groups=self.nbands,
                                   kernel_size=kernel.shape,
                                   padding='same',
                                   bias=False)

        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        self.loss = nn.L1Loss(reduction='sum')

        # Mask definition
        if mask is not None:
            self.mask = mask
        else:
            self.mask = torch.ones((1, self.nbands, pan_shape[-2],
                                    pan_shape[-1]), device=self.device)
            
    def forward(self, outputs, labels):

        x = self.depthconv(outputs)

        #labels = labels[:, :, self.pad:-self.pad, self.pad:-self.pad]
        
        y = torch.zeros(x.shape, device=self.device)
        W_ = torch.zeros(x.shape, device=self.device)

        for b in range(self.nbands):
            y[:, b, self.MTF_r::self.ratio, self.MTF_c::self.ratio] = labels[:, b, 2::self.ratio, 2::self.ratio]
            W_[:, b, self.MTF_r::self.ratio, self.MTF_c::self.ratio] = self.mask[:, b, 2::self.ratio, 2::self.ratio]

        W_ = W_ / torch.sum(W_)

        x = x * W_
        y = y * W_
        L = self.loss(x, y)

        return L


class StructuralLoss(nn.Module):

    def __init__(self, sigma, device):
        # Class initialization
        super(StructuralLoss, self).__init__()

        # Parameters definition:

        self.scale = ceil(sigma / 2)
        self.device = device

    def forward(self, outputs, labels, xcorr_thr):
        X_corr = torch.clamp(ccorr(outputs, labels, self.scale, self.device), min=-1)
        X = 1.0 - X_corr

        with torch.no_grad():
            Lxcorr_no_weights = torch.mean(X)

        worst = X.gt(xcorr_thr)
        Y = X * worst
        Lxcorr = torch.mean(Y)

        return Lxcorr, Lxcorr_no_weights.item()
    
class SpectralStructuralLoss(nn.Module):
    '''
        Combination of StrucuralLoss and SpectralLoss
    '''

    def __init__(self, img_pan_shape, device, sensor):
        super().__init__()
        self.device = device
        self.beta = sensor.beta
        self.sensor = sensor
        self.spec_loss = SpectralLossNocorr(gen_mtf(sensor.ratio,
                                                    sensor.sensor),
                                            sensor.net_scope,
                                            img_pan_shape,
                                            sensor.ratio,
                                            self.device)
        self.struct_loss = StructuralLoss(sensor.ratio, self.device)

    def forward(self, outputs, labels, inp):
        labels_spec = labels[:,:-1,:,:]
        labels_struct = labels[:,-1,:,:].unsqueeze(dim=1)
        threshold = local_corr_mask(inp.to('cpu'),
                                    self.sensor.ratio,
                                    self.sensor.sensor,
                                    self.device,
                                    8).float()
        
        return self.spec_loss(outputs, labels_spec) + self.beta * self.struct_loss(outputs, labels_struct, threshold)[0]