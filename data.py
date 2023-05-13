import os

import numpy as np
import torch
from scipy.io import loadmat
from skimage.transform import rescale
from torch.utils.data import Dataset

class PAN_Dataset(Dataset):
    
    def __init__(self, images_dir, sensor, full_resolution=False, eval=False):
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.sensor = sensor
        self.full_resolution = full_resolution
        self.eval = eval

    def __getitem__(self, index):
        '''
            Loads and returns a sample from the dataset at the given index.

            For full-resolution training, returns the MS image upsampled, stacked with the PAN image at full-resolution as input, same for the label
            For full-resolution training, returns the MS image upsampled, stacked with the PAN image at full-resolution as input, MS and PAN images as labels
            For reduced-resolution training/evaluation, returns  I_in generated using Wald's protocol, and the MS image as label
        '''
        image = loadmat(self.image_paths[index])

        if self.full_resolution:
            I_PAN, I_MS = image['I_PAN'], image['I_MS']
            I_MS_UP = rescale(I_MS, [self.sensor.ratio, self.sensor.ratio, 1], order=3)
            I_MS_UP = np.moveaxis(I_MS_UP, -1, 0) / (2 ** self.sensor.nbits)
            I_PAN = I_PAN[None, :, :] / (2 ** self.sensor.nbits)
            I_MS_UP = torch.tensor(I_MS_UP, dtype=torch.float32)
            I_PAN = torch.tensor(I_PAN, dtype=torch.float32)
            if self.eval:
                I_MS = np.moveaxis(I_MS, -1, 0) / (2 ** self.sensor.nbits)
                I_MS = torch.tensor(I_MS, dtype=torch.float32)
                return torch.cat([I_MS_UP, I_PAN], dim=0), I_MS, I_PAN
            return torch.cat([I_MS_UP, I_PAN], dim=0), torch.cat([I_MS_UP, I_PAN], dim=0)
        else:
            I_in, I_MS = image['I_in'], image['I_MS']
            I_in = np.moveaxis(I_in, -1, 0) / (2 ** self.sensor.nbits)
            I_MS = np.moveaxis(I_MS, -1, 0) / (2 ** self.sensor.nbits)
            I_in = torch.tensor(I_in, dtype=torch.float32)
            I_MS = torch.tensor(I_MS, dtype=torch.float32)
            return I_in, I_MS

    def __len__(self):
        return len(self.image_paths)