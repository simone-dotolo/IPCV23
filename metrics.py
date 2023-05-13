from math import ceil, floor, log2

import numpy as np
from scipy import ndimage as ndi

from cross_correlation import xcorr
from utils import resize_with_mtf

def SAM(predictions, labels):
    '''
        Spectral Angle Mapper (SAM).

        Args:
            predictions:  Numpy Array
                The Fused image. Dimensions: H, W, Bands
            labels : Numpy Array
                The reference image. Dimensions: H, W, Bands

        Returns:
            angle : float
                The SAM index in degree.
    '''
    
    preds_norm = np.sum(predictions ** 2, axis=-1)
    labels_norm = np.sum(labels ** 2, axis=-1)
    product_norm = np.sqrt(preds_norm * labels_norm)
    preds_labels_prod = np.sum(predictions * labels, axis=-1)
    preds_labels_prod[preds_labels_prod == 0] = np.nan
    product_norm[product_norm == 0] = np.nan
    preds_labels_prod = preds_labels_prod.flatten()
    product_norm = product_norm.flatten()
    angle = np.mean(np.arccos(np.clip(preds_labels_prod / product_norm, a_min=-1, a_max=1)))
    angle = angle * 180 / np.pi
    return angle


def ERGAS(predictions, labels, ratio):
    '''
        Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS).
        
        Args:
            predictions : Numpy Array
                The Fused image. Dimensions: H, W, Bands
            labels : Numpy Array
                The reference image. Dimensions: H, W, Bands
            ratio : int
                PAN-MS resolution ratio

        Returns:
            ergas_index : float
                The ERGAS index.
    '''

    nbands = labels.shape[-1]
    mu = np.mean(labels, axis=(0, 1)) ** 2
    error = np.mean((predictions - labels) ** 2, axis=(0, 1))
    ergas = 100 / ratio * np.sqrt(np.sum(error / (mu * nbands), axis=-1))
    return ergas

def Q(predictions, labels, block_size=32):
    '''
        Universal Image Quality Index (UIQI).

        Args:
            predictions : Numpy Array
                The Fused image. Dimensions: H, W, Bands
            labels : Numpy Array
                The reference image. Dimensions: H, W, Bands
            block_size : int
                The windows size on which calculate the Q index

        Returns:
            quality : float
                The UIQI index.
    '''

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
        quality_map = np.ones(p_sum.shape)
        indexes = ((var1 + var2) == 0) & ((mu1 ** 2 + mu2 ** 2) != 0)
        quality_map[indexes] = (2 * mu1[indexes] * mu2[indexes]) / (mu1[indexes] ** 2 + mu2[indexes] ** 2)
        indexes = ((var1 + var2) != 0) & ((mu1 ** 2 + mu2 ** 2) == 0)
        quality_map[indexes] = (2 * cov12[indexes]) / (var1[indexes] + var2[indexes])
        indexes = ((var1 + var2) != 0) & ((mu1 ** 2 + mu2 ** 2) != 0)
        quality_map[indexes] = (4 * cov12[indexes] * mu1[indexes] * mu2[indexes]) / (var1[indexes] + var2[indexes]) / (mu1[indexes] ** 2 + mu2[indexes] ** 2)
        q_indexes[i] = np.mean(quality_map, axis=(0,1))
    
    return np.mean(q_indexes).item()

'''
    Reference: https://github.com/matciotola/fr-pansh-eval-tool/blob/main/NumPy/metrics.py
'''

def _normalize_block(im):
    """
        Auxiliary Function for Q2n computation.

        Parameters
        ----------
        im : Numpy Array
            Image on which calculate the statistics. Dimensions: H, W

        Return
        ------
        y : Numpy array
            The normalized version of im
        m : float
            The mean of im
        s : float
            The standard deviation of im

    """

    m = np.mean(im)
    s = np.std(im, ddof=1)

    if s == 0:
        s = 1e-10

    y = ((im - m) / s) + 1

    return y, m, s


def _cayley_dickson_property_1d(onion1, onion2):
    """
        Cayley-Dickson construction for 1-D arrays.
        Auxiliary function for Q2n calculation.

        Parameters
        ----------
        onion1 : Numpy Array
            First 1-D array
        onion2 : Numpy Array
            Second 1-D array

        Return
        ------
        ris : Numpy array
            The result of Cayley-Dickson construction on the two arrays.
    """

    n = onion1.__len__()

    if n > 1:
        half_pos = int(n / 2)
        a = onion1[:half_pos]
        b = onion1[half_pos:]

        neg = np.ones(b.shape)
        neg[1:] = -1

        b = b * neg
        c = onion2[:half_pos]
        d = onion2[half_pos:]
        d = d * neg

        if n == 2:
            ris = np.concatenate([(a * c) - (d * b), (a * d) + (c * b)])
        else:
            ris1 = _cayley_dickson_property_1d(a, c)

            ris2 = _cayley_dickson_property_1d(d, b * neg)
            ris3 = _cayley_dickson_property_1d(a * neg, d)
            ris4 = _cayley_dickson_property_1d(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = np.concatenate([aux1, aux2])
    else:
        ris = onion1 * onion2

    return ris


def _cayley_dickson_property_2d(onion1, onion2):
    """
        Cayley-Dickson construction for 2-D arrays.
        Auxiliary function for Q2n calculation.

        Parameters
        ----------
        onion1 : Numpy Array
            First MultiSpectral img. Dimensions: H, W, Bands
        onion2 : Numpy Array
            Second MultiSpectral img. Dimensions: H, W, Bands

        Return
        ------
        ris : Numpy array
            The result of Cayley-Dickson construction on the two arrays.
    """

    dim3 = onion1.shape[-1]
    if dim3 > 1:
        half_pos = int(dim3 / 2)

        a = onion1[:, :, :half_pos]
        b = onion1[:, :, half_pos:]
        b = np.concatenate([np.expand_dims(b[:, :, 0], -1), -b[:, :, 1:]], axis=-1)

        c = onion2[:, :, :half_pos]
        d = onion2[:, :, half_pos:]
        d = np.concatenate([np.expand_dims(d[:, :, 0], -1), -d[:, :, 1:]], axis=-1)

        if dim3 == 2:
            ris = np.concatenate([(a * c) - (d * b), (a * d) + (c * b)], axis=-1)
        else:
            ris1 = _cayley_dickson_property_2d(a, c)
            ris2 = _cayley_dickson_property_2d(d,
                                              np.concatenate([np.expand_dims(b[:, :, 0], -1), -b[:, :, 1:]], axis=-1))
            ris3 = _cayley_dickson_property_2d(np.concatenate([np.expand_dims(a[:, :, 0], -1), -a[:, :, 1:]], axis=-1),
                                              d)
            ris4 = _cayley_dickson_property_2d(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = np.concatenate([aux1, aux2], axis=-1)
    else:
        ris = onion1 * onion2

    return ris


def _q_index_metric(im1, im2, size):
    """
        Q2n calculation on a window of dimension (size, size).
        Auxiliary function for Q2n calculation.

        Parameters
        ----------
        im1 : Numpy Array
            First MultiSpectral img. Dimensions: H, W, Bands
        im2 : Numpy Array
            Second MultiSpectral img. Dimensions: H, W, Bands
        size : int
            The size of the squared windows on which calculate the UQI index


        Return
        ------
        q : Numpy array
            The Q2n calculated on a window of dimension (size,size).
    """

    im1 = im1.astype(np.double)
    im2 = im2.astype(np.double)
    im2 = np.concatenate([np.expand_dims(im2[:, :, 0], -1), -im2[:, :, 1:]], axis=-1)

    depth = im1.shape[-1]
    for i in range(depth):
        im1[:, :, i], m, s = _normalize_block(im1[:, :, i])
        if m == 0:
            if i == 0:
                im2[:, :, i] = im2[:, :, i] - m + 1
            else:
                im2[:, :, i] = -(-im2[:, :, i] - m + 1)
        else:
            if i == 0:
                im2[:, :, i] = ((im2[:, :, i] - m) / s) + 1
            else:
                im2[:, :, i] = -(((-im2[:, :, i] - m) / s) + 1)

    m1 = np.mean(im1, axis=(0, 1))
    m2 = np.mean(im2, axis=(0, 1))

    mod_q1m = np.sqrt(np.sum(m1 ** 2))
    mod_q2m = np.sqrt(np.sum(m2 ** 2))

    mod_q1 = np.sqrt(np.sum(im1 ** 2, axis=-1))
    mod_q2 = np.sqrt(np.sum(im2 ** 2, axis=-1))

    term2 = mod_q1m * mod_q2m
    term4 = mod_q1m ** 2 + mod_q2m ** 2
    temp = (size ** 2) / (size ** 2 - 1)
    int1 = temp * np.mean(mod_q1 ** 2)
    int2 = temp * np.mean(mod_q2 ** 2)
    int3 = temp * (mod_q1m ** 2 + mod_q2m ** 2)
    term3 = int1 + int2 - int3

    mean_bias = 2 * term2 / term4

    if term3 == 0:
        q = np.zeros((1, 1, depth), dtype='float64')
        q[:, :, -1] = mean_bias
    else:
        cbm = 2 / term3
        qu = _cayley_dickson_property_2d(im1, im2)
        qm = _cayley_dickson_property_1d(m1, m2)

        qv = temp * np.mean(qu, axis=(0, 1))
        q = qv - temp * qm
        q = q * mean_bias * cbm

    return q


def Q2n(outputs, labels, q_block_size=32, q_shift=32):
    """
        Q2n calculation on a window of dimension (size, size).
        Auxiliary function for Q2n calculation.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144
        [Garzelli09]        A. Garzelli and F. Nencini, "Hypercomplex quality assessment of multi/hyper-spectral images,"
                            IEEE Geoscience and Remote Sensing Letters, vol. 6, no. 4, pp. 662-665, October 2009.
        [Vivone20]          G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods",
                            IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.

        Parameters
        ----------
        outputs : Numpy Array
            The Fused image. Dimensions: H, W, Bands
        labels : Numpy Array
            The reference image. Dimensions: H, W, Bands
        q_block_size : int
            The windows size on which calculate the Q2n index
        q_shift : int
            The stride for Q2n index calculation

        Return
        ------
        q2n_index : float
            The Q2n index.
        q2n_index_map : Numpy Array
            The Q2n map, on a support of (q_block_size, q_block_size)
    """

    height, width, depth = labels.shape
    stepx = ceil(height / q_shift)
    stepy = ceil(width / q_shift)

    if stepy <= 0:
        stepx = 1
        stepy = 1

    est1 = (stepx - 1) * q_shift + q_block_size - height
    est2 = (stepy - 1) * q_shift + q_block_size - width

    if (est1 != 0) and (est2 != 0):
        labels = np.pad(labels, ((0, est1), (0, est2), (0, 0)), mode='reflect')
        outputs = np.pad(outputs, ((0, est1), (0, est2), (0, 0)), mode='reflect')

        outputs = outputs.astype(np.int16)
        labels = labels.astype(np.int16)

    height, width, depth = labels.shape

    if ceil(log2(depth)) - log2(depth) != 0:
        exp_difference = 2 ** (ceil(log2(depth))) - depth
        diff_zeros = np.zeros((height, width, exp_difference), dtype="float64")
        labels = np.concatenate([labels, diff_zeros], axis=-1)
        outputs = np.concatenate([outputs, diff_zeros], axis=-1)

    height, width, depth = labels.shape

    values = np.zeros((stepx, stepy, depth))
    for j in range(stepx):
        for i in range(stepy):
            values[j, i, :] = _q_index_metric(
                labels[j * q_shift:j * q_shift + q_block_size, i * q_shift: i * q_shift + q_block_size, :],
                outputs[j * q_shift:j * q_shift + q_block_size, i * q_shift: i * q_shift + q_block_size, :],
                q_block_size
            )

    q2n_index_map = np.sqrt(np.sum(values ** 2, axis=-1))
    q2n_index = np.mean(q2n_index_map)

    return q2n_index.item(), q2n_index_map

def ReproERGAS(outputs, ms, pan, sensor, ratio=4, dim_cut=0):
    """
        Reprojected Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS).

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
        R-ERGAS : float
            The R-ERGAS index

    """

    outputs, ms = resize_with_mtf(outputs, ms, pan, sensor, ratio, dim_cut)
    return ERGAS(outputs, ms, ratio)


def ReproSAM(outputs, ms, pan, sensor, ratio=4, dim_cut=0):
    """
        Reprojected Spectral Angle Mapper (SAM).

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
        R-SAM : float
            The R-SAM index

    """
    outputs, ms = resize_with_mtf(outputs, ms, pan, sensor, ratio, dim_cut)
    return SAM(outputs, ms)


def ReproQ2n(outputs, ms, pan, sensor, ratio=4, q_block_size=32, q_shift=32, dim_cut=0):
    """
        Reprojected Q2n.

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
        q_block_size : int
            The windows size on which calculate the Q2n index
        q_shift : int
            The stride for Q2n index calculation
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        r_q2n : float
            The R-Q2n index

    """
    outputs, ms = resize_with_mtf(outputs, ms, pan, sensor, ratio, dim_cut)
    r_q2n, _ = Q2n(outputs, ms, q_block_size, q_shift)
    return r_q2n


def ReproQ(outputs, ms, pan, sensor, ratio=4, q_block_size=32, dim_cut=0):
    """
        Reprojected Q.

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
        q_block_size : int
            The windows size on which calculate the Q index
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        r_q : float
            The R-Q index

    """
    outputs, ms = resize_with_mtf(outputs, ms, pan, sensor, ratio, dim_cut)
    r_q = Q(outputs, ms, q_block_size)
    return r_q


def ReproMetrics(outputs, ms, pan, sensor, ratio=4, q_block_size=32, q_shift=32, dim_cut=0):
    """
        Computation of all reprojected metrics.

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
        q_block_size : int
            The windows size on which calculate the Q2n and Q index
        q_shift : int
            The stride for Q2n index calculation
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        r_q2n : float
            The R-Q2n index
        r_q : float
            The R-Q index
        R-SAM : float
            The R-SAM index
        R-ERGAS : float
            The R-ERGAS index

    """
    outputs, ms = resize_with_mtf(outputs, ms, pan, sensor, ratio, dim_cut)
    q2n, _ = Q2n(outputs, ms, q_block_size, q_shift)
    q = Q(outputs, ms, q_block_size)
    sam = SAM(outputs, ms)
    ergas = ERGAS(outputs, ms, ratio)
    return q2n, q, sam, ergas


def DRho(outputs, pan, sigma=4):
    """
        Spatial Quality Index based on local cross-correlation.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sigma : int
            The windows size on which calculate the Drho index; Accordingly with the paper it should be the
            resolution scale which elapses between MS and PAN.

        Return
        ------
        d_rho : float
            The d_rho index

    """
    half_width = ceil(sigma / 2)
    rho = np.clip(xcorr(outputs, pan, half_width), a_min=-1.0, a_max=1.0)
    d_rho = 1.0 - rho
    return np.mean(d_rho).item()