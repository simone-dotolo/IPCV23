from math import floor, log

import numpy as np
import torch
from scipy import ndimage as ndi
from torch import nn

from cross_correlation import xcorr, xcorr_torch
from spectral_tools import gen_mtf

def interp23tap(img, ratio):
    """
        Polynomial (with 23 coefficients) interpolator Function.

        For more details please refers to:

        [1]  B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli - Context-driven fusion of high spatial and spectral
             resolution images based on oversampled multiresolution analysis
        [2] B. Aiazzi, S. Baronti, M. Selva, and L. Alparone - Bi-cubic interpolation for shift-free pan-sharpening
        [3] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. Orn Ulfarsson, L. Alparone, J. Chanussot -
            A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with
            classical and emerging pansharpening methods


        Parameters
        ----------
        img : Numpy Array
            Image to be scaled. Dimension: H, W, B
        ratio : int
            The desired scale. It must be a factor power of 2.


        Return
        ------
        img : Numpy array
            the interpolated img.

        """
    assert ((2 ** (round(log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'

    r, c, b = img.shape

    CDF23 = np.asarray(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
    CDF23 = [element * 2 for element in CDF23]
    BaseCoeff = np.expand_dims(np.concatenate([np.flip(CDF23[1:]), CDF23]), axis=-1)

    for z in range(int(ratio / 2)):

        I1LRU = np.zeros(((2 ** (z + 1)) * r, (2 ** (z + 1)) * c, b))

        if z == 0:
            I1LRU[1::2, 1::2, :] = img
        else:
            I1LRU[::2, ::2, :] = img

        for i in range(b):
            temp = ndi.convolve(np.transpose(I1LRU[:, :, i]), BaseCoeff, mode='wrap')
            I1LRU[:, :, i] = ndi.convolve(np.transpose(temp), BaseCoeff, mode='wrap')

        img = I1LRU

    return img

def coregistration(ms, pan, kernel, ratio=4, search_win=4):
    """
        Coregitration function for MS-PAN pair.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        ms : Numpy Array
            The Multi-Spectral image. Dimensions: H, W, Bands
        pan : Numpy Array
            The PAN image. Dimensions: H, W
        kernel : Numpy Array
            The filter array.
        ratio : int
            PAN-MS resolution ratio
        search_win : int
            The windows in which search the optimal value for the coregistration step

        Return
        ------
        r : Numpy Array
            The optimal raw values.
        c : Numpy Array
            The optimal column values.
    """

    nbands = ms.shape[-1]
    p = ndi.convolve(pan, kernel, mode='nearest')
    rho = np.zeros((search_win, search_win, nbands))
    r = np.zeros(nbands)
    c = np.copy(r)

    for i in range(search_win):
        for j in range(search_win):
            rho[i, j, :] = np.mean(
                xcorr(ms, np.expand_dims(p[i::ratio, j::ratio], -1), floor(ratio / 2)), axis=(0, 1))

    max_value = np.amax(rho, axis=(0, 1))

    for b in range(nbands):
        x = rho[:, :, b]
        max_value = x.max()
        pos = np.where(x == max_value)
        if len(pos[0]) != 1:
            pos = (pos[0][0], pos[1][0])
        pos = tuple(map(int, pos))
        r[b] = pos[0]
        c[b] = pos[1]
        r = np.squeeze(r).astype(np.uint8)
        c = np.squeeze(c).astype(np.uint8)

    return r, c


def resize_with_mtf(outputs, ms, pan, sensor, ratio=4, dim_cut=21):
    """
        Resize of Fused Image to MS scale, in according to the coregistration with the PAN.
        If dim_cut is different by zero a cut is made on both outputs and ms, to discard possibly values affected by paddings.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        ms : Numpy Array
            MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sensor : str
            The name of the satellites which has provided the images.
        ratio : int
            Resolution scale which elapses between MS and PAN.
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        x : NumPy array
            Fused MultiSpectral image, coregistered with the PAN, low-pass filtered and decimated. If dim_cut is different
            by zero it is also cut
        ms : NumPy array
            MultiSpectral img. If dim_cut is different by zero it is cut.
    """

    kernel = gen_mtf(ratio, sensor)
    kernel = kernel.astype(np.float32)
    nbands = kernel.shape[-1]
    pad_size = floor((kernel.shape[0] - 1) / 2)

    r, c = coregistration(ms, pan, kernel[:, :, 0], ratio)

    kernel = np.moveaxis(kernel, -1, 0)
    kernel = np.expand_dims(kernel, axis=1)

    kernel = torch.from_numpy(kernel).type(torch.float32)

    depthconv = nn.Conv2d(in_channels=nbands,
                          out_channels=nbands,
                          groups=nbands,
                          kernel_size=kernel.shape,
                          bias=False)
    depthconv.weight.data = kernel
    depthconv.weight.requires_grad = False
    pad = nn.ReplicationPad2d(pad_size)

    x = np.zeros(ms.shape, dtype=np.float32)

    outputs = np.expand_dims(np.moveaxis(outputs, -1, 0), 0)
    outputs = torch.from_numpy(outputs)

    outputs = pad(outputs)
    outputs = depthconv(outputs)

    outputs = outputs.detach().cpu().numpy()
    outputs = np.moveaxis(np.squeeze(outputs, 0), 0, -1)

    for b in range(nbands):
        x[:, :, b] = outputs[r[b]::ratio, c[b]::ratio, b]

    if dim_cut != 0:
        x = x[dim_cut:-dim_cut, dim_cut:-dim_cut, :]
        ms = ms[dim_cut:-dim_cut, dim_cut:-dim_cut, :]

    return x, ms


def net_scope(kernel_size):
    """
        Compute the network scope.

        Parameters
        ----------
        kernel_size : List[int]
            A list containing the kernel size of each layer of the network.

        Return
        ------
        scope : int
            The scope of the network

        """

    scope = 0
    for i in range(len(kernel_size)):
        scope += floor(kernel_size[i] / 2)
    return scope

def local_corr_mask(img_in, ratio, sensor, device, kernel=8):
    """
        Compute the threshold mask for the structural loss.

        Parameters
        ----------
        img_in : Torch Tensor
            The test image, already normalized and with the MS part upsampled with ideal interpolator.
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        device : Torch device
            The device on which perform the operation.
        kernel : int
            The semi-width for local cross-correlation computation.
            (See the cross-correlation function for more details)

        Return
        ------
        mask : PyTorch Tensor
            Local correlation field stack, composed by each MS and PAN. Dimensions: Batch, B, H, W.

        """

    I_PAN = torch.unsqueeze(img_in[:, -1, :, :], dim=1)
    I_MS = img_in[:, :-1, :, :]

    MTF_kern = gen_mtf(ratio, sensor)[:, :, 0]
    MTF_kern = np.expand_dims(MTF_kern, axis=(0, 1))
    MTF_kern = torch.from_numpy(MTF_kern).type(torch.float32)
    pad = floor((MTF_kern.shape[-1] - 1) / 2)

    padding = nn.ReflectionPad2d(pad)

    depthconv = nn.Conv2d(in_channels=1,
                            out_channels=1,
                            groups=1,
                            kernel_size=MTF_kern.shape,
                            bias=False)

    depthconv.weight.data = MTF_kern
    depthconv.weight.requires_grad = False

    I_PAN = padding(I_PAN)
    I_PAN = depthconv(I_PAN)
    mask = xcorr_torch(I_PAN, I_MS, kernel, device)
    mask = 1.0 - mask

    return mask