from math import floor, ceil
from spectral_tools import generate_mtf_variables
import numpy as np
import torch
import torch.nn as nn
from utils import xcorr_torch as ccorr

class SpectralLoss(nn.Module):
    def __init__(self, mtf, ratio, device):
        # Class initialization
        super(SpectralLoss, self).__init__()

        # Parameters definition
        kernel = mtf
        self.nbands = kernel.shape[-1]
        self.ratio = ratio
        self.device = device
        # Conversion of filters in Tensor
        self.pad_x = floor((kernel.shape[0] - 1) / 2)
        self.pad_y = floor((kernel.shape[1] - 1) / 2)
        kernel = np.moveaxis(kernel, -1, 0)
        kernel = np.expand_dims(kernel, axis=1)

        kernel = torch.from_numpy(kernel).type(torch.float32)

        # DepthWise-Conv2d definition
        self.depthconv = nn.Conv2d(in_channels=self.nbands,
                                   out_channels=self.nbands,
                                   groups=self.nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)

        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        self.loss = nn.L1Loss(reduction='sum')

    def forward(self, outputs, labels, r, c):
        x = self.depthconv(outputs)
        labels = labels[:, :, self.pad_x:-self.pad_x, self.pad_y:-self.pad_y]
        y = torch.zeros(x.shape, device=self.device)
        W_ = torch.zeros(x.shape, device=self.device)
        mask = torch.ones(x.shape, device=self.device)

        for b in range(self.nbands):
            y[:, b, r[b]::self.ratio, c[b]::self.ratio] = labels[:, b, 2::self.ratio, 2::self.ratio]
            W_[:, b, r[b]::self.ratio, c[b]::self.ratio] = mask[:, b, 2::self.ratio, 2::self.ratio]

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

    # Linear combination of SpectralLoss and StructuralLoss

    def __init__(self, img_pan, img_ms, device, sensor):
        super().__init__()
        self.device = device
        self.beta = sensor.beta
        self.img_pan = img_pan
        self.img_ms = img_ms
        self.spec_loss = SpectralLoss(generate_mtf_variables(sensor.ratio,
                                                             sensor,
                                                             self.img_pan,
                                                             self.img_ms),
                                      sensor.ratio,
                                      self.device)
        self.struct_loss = StructuralLoss(sensor.ratio, self.device) 

    def forward(self, outputs, labels):

        labels_spec = labels[:,:-1,:,:]
        labels_struct = labels[:,-1,:,:].unsqueeze(dim=1)
        
        return self.spec_loss(outputs, labels_spec) + self.beta * self.struct_loss(outputs, labels_struct)[0]
        #return self.struct_loss(outputs, labels_struct, 0)[0]