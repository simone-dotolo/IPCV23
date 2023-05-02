import numpy as np
from math import floor
from scipy import ndimage as ndi

def SAM(predictions, labels):
    
    preds_norm = np.sum(predictions ** 2, axis=-1)
    labels_norm = np.sum(labels ** 2, axis=-1)
    product_norm = np.sqrt(preds_norm * labels_norm)
    preds_labels_prod = np.sum(predictions * labels, axis=-1)
    preds_labels_prod[preds_labels_prod == 0] = np.nan
    product_norm[product_norm == 0] = np.nan
    preds_labels_prod = preds_labels_prod.flatten()
    product_norm = product_norm.flatten()
    # TODO: maybe replace np.sum() / product_norm.shape[0] with np.mean()
    angle = np.sum(np.arccos(np.clip(preds_labels_prod / product_norm, a_min=-1, a_max=1)), axis=-1) / product_norm.shape[0]
    angle = angle * 180 / np.pi
    return angle


def ERGAS(predictions, labels, ratio):
    
    nbands = labels.shape[-1]
    mu = np.mean(labels, axis=(0, 1)) ** 2
    error = np.mean((predictions - labels) ** 2, axis=(0, 1))
    ergas = 100 / ratio * np.sqrt(np.sum(error / (mu * nbands), axis=-1))
    return ergas

def Q(predictions, labels, block_size=32):

    N = block_size ** 2
    nbands = predictions.shape[-1]
    kernel = np.ones((block_size, block_size))

    valid_pad = floor((block_size - 1) / 2)

    q_indexes = np.ones(nbands)

    for i in range(nbands):
        p_sum = ndi.convolve(predictions[:, :, i], kernel)[valid_pad:-valid_pad, valid_pad:-valid_pad]
        l_sum = ndi.convolve(labels[:, :, i], kernel)[valid_pad:-valid_pad, valid_pad:-valid_pad]
        p_sq_sum = ndi.convolve(predictions[:, :, i] ** 2, kernel)[valid_pad:-valid_pad, valid_pad:-valid_pad]
        l_sq_sum = ndi.convolve(labels[:, :, i] ** 2, kernel)[valid_pad:-valid_pad, valid_pad:-valid_pad]
        p_l_sum = ndi.convolve(predictions[:, :, i] * labels[:, :, i], kernel)[valid_pad:-valid_pad, valid_pad:-valid_pad]
        mu1 = p_sum / N
        mu2 = l_sum / N
        var1 = p_sq_sum / N - mu1 ** 2
        var2 = l_sq_sum / N - mu2 ** 2
        cov12 = p_l_sum / N - mu1 * mu2
        indexes = ((var1 + var2) != 0) & ((mu1 ** 2 + mu2 ** 2) != 0)
        quality_map = np.ones(p_sum.shape)
        quality_map[indexes] = (4 * cov12[indexes] * mu1[indexes] * mu2[indexes]) / (var1[indexes] + var2[indexes]) / (mu1[indexes] ** 2 + mu2[indexes] ** 2)
        q_indexes[i] = np.mean(quality_map, axis=(0,1))
    
    return np.mean(q_indexes).item()

def Q2n():
    pass