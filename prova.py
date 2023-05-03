from math import ceil
from timeit import default_timer as timer
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

def xcorr_torch(img_1, img_2, half_width, device):
    """
        A PyTorch implementation of Cross-Correlation Field computation.
        Parameters
        ----------
        img_1 : Torch Tensor
            First image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
        img_2 : Torch Tensor
            Second image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
        half_width : int
            The semi-size of the window on which calculate the cross-correlation
        device : Torch Device
            The device on which perform the operation.
        Return
        ------
        L : Torch Tensor
            The cross-correlation map between img_1 and img_2
    """

    img_1 = torch.tensor(img_1)
    img_2 = torch.tensor(img_2)

    w = ceil(half_width)
    ep = 1e-20
    img_1 = img_1.type(torch.DoubleTensor)
    img_2 = img_2.type(torch.DoubleTensor)

    img_1 = img_1.to(device)
    img_2 = img_2.to(device)

    img_1 = F.pad(img_1, (w, w, w, w))
    img_2 = F.pad(img_2, (w, w, w, w))

    img_1_cum = torch.cumsum(torch.cumsum(img_1, dim=-1), dim=-2)
    img_2_cum = torch.cumsum(torch.cumsum(img_2, dim=-1), dim=-2)

    img_1_mu = (img_1_cum[:, :, 2 * w:, 2 * w:] - img_1_cum[:, :, :-2 * w, 2 * w:] - img_1_cum[:, :, 2 * w:, :-2 * w] +
                img_1_cum[:, :, :-2 * w, :-2 * w]) / (4 * w ** 2)
    img_2_mu = (img_2_cum[:, :, 2 * w:, 2 * w:] - img_2_cum[:, :, :-2 * w, 2 * w:] - img_2_cum[:, :, 2 * w:, :-2 * w] +
                img_2_cum[:, :, :-2 * w, :-2 * w]) / (4 * w ** 2)

    img_1 = img_1[:, :, w:-w, w:-w] - img_1_mu
    img_2 = img_2[:, :, w:-w, w:-w] - img_2_mu

    img_1 = F.pad(img_1, (w, w, w, w))
    img_2 = F.pad(img_2, (w, w, w, w))

    i2_cum = torch.cumsum(torch.cumsum(img_1 ** 2, dim=-1), dim=-2)
    j2_cum = torch.cumsum(torch.cumsum(img_2 ** 2, dim=-1), dim=-2)
    ij_cum = torch.cumsum(torch.cumsum(img_1 * img_2, dim=-1), dim=-2)

    sig2_ij_tot = (ij_cum[:, :, 2 * w:, 2 * w:] - ij_cum[:, :, :-2 * w, 2 * w:] - ij_cum[:, :, 2 * w:, :-2 * w] +
                   ij_cum[:, :, :-2 * w, :-2 * w])
    sig2_ii_tot = (i2_cum[:, :, 2 * w:, 2 * w:] - i2_cum[:, :, :-2 * w, 2 * w:] - i2_cum[:, :, 2 * w:, :-2 * w] +
                   i2_cum[:, :, :-2 * w, :-2 * w])
    sig2_jj_tot = (j2_cum[:, :, 2 * w:, 2 * w:] - j2_cum[:, :, :-2 * w, 2 * w:] - j2_cum[:, :, 2 * w:, :-2 * w] +
                   j2_cum[:, :, :-2 * w, :-2 * w])

    sig2_ii_tot = torch.clip(sig2_ii_tot, ep, sig2_ii_tot.max().item())
    sig2_jj_tot = torch.clip(sig2_jj_tot, ep, sig2_jj_tot.max().item())

    L = sig2_ij_tot / ((sig2_ii_tot * sig2_jj_tot) ** 0.5 + ep)

    return torch.mean(1.0 - torch.clip(L, min=-1.0))

def corr(a, b, sigma):

    N = sigma ** 2
    eps = 1e-20
    kernel = np.ones(shape=(1,1,sigma, sigma))

    mu1 = ndi.convolve(a, kernel) / N
    mu2 = ndi.convolve(b, kernel) / N
    var1 = ndi.convolve(a ** 2, kernel) / N - mu1 ** 2
    var2 = ndi.convolve(b ** 2, kernel) / N - mu2 ** 2
    cov12 = ndi.convolve(a * b, kernel) / N - mu1 * mu2

    cc = np.ones(cov12.shape)
    indexes = (var1 != 0) | (var2 != 0)
    cc[indexes] = cov12[indexes] / ((var1[indexes] * var2[indexes]) ** 0.5 + eps) 

    return np.mean(1.0 - np.clip(cc, a_min=-1.0, a_max=1.0))


a = np.ones(shape=(1,3,512,512))
b = np.ones(shape=(1,3,512,512)) * 2

start = timer()
res1 = xcorr_torch(a, b, 4, 'cpu')
end = timer()
print(res1.item(), end - start)

start = timer()
res2 = corr(a, b, 8)
end = timer()
print(res2, end - start)