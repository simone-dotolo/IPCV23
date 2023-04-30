import numpy as np

def SAM(predictions, labels):
    
    preds_norm = np.sum(predictions ** 2, axis=-1)
    labels_norm = np.sum(labels ** 2, axis=-1)
    product_norm = np.sqrt(preds_norm * labels_norm)
    preds_labels_prod = np.sum(predictions * labels, axis=-1)
    preds_labels_prod[preds_labels_prod == 0] = np.nan
    product_norm[product_norm == 0] = np.nan
    preds_labels_prod = preds_labels_prod.flatten()
    product_norm = product_norm.flatten()
    angle = np.sum(np.arccos(np.clip(preds_labels_prod / product_norm, a_min=-1, a_max=1)), axis=-1) / product_norm.shape[0]
    angle = angle * 180 / np.pi
    return angle


def ERGAS(predictions, labels, ratio):
    
    bands = labels.shape[-1]
    mu = np.mean(labels, axis=(0, 1)) ** 2
    error = np.mean((predictions - labels) ** 2, axis=(0, 1))
    ergas = 100 / ratio * np.sqrt(np.sum(error / (mu * bands), axis=-1))
    return ergas

def Q2n(predictions, labels, block_size=32, stride=32):
    pass

def SpectralLoss():
    pass

def SpatialLoss():
    pass

def SpatialCorrLoss():
    pass