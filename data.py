import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.io import loadmat
from skimage.transform import rescale

class PAN_Dataset(Dataset):
    
    def __init__(self, images_dir, sensor, full_resolution=False):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.sensor = sensor
        self.full_resolution = full_resolution

    def transform(self, I_in, I_PAN, I_MS):

        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x : x.type(torch.float32) / (2 ** self.sensor.nbits))
        ])

        if self.full_resolution is False:
            X, y = transf(I_in), transf(I_MS)
            return X, y

        I_MS_UP = rescale(I_MS, [self.sensor.ratio, self.sensor.ratio, 1], order=3)
        I_PAN = np.expand_dims(I_PAN, axis=-1)

        X = np.concatenate([I_MS_UP, I_PAN], axis=-1)
        
        return transf(X), transf(X)

    def __getitem__(self, index):
        
        image = loadmat(self.image_paths[index])

        I_in, I_PAN, I_MS = image['I_in'], image['I_PAN'], image['I_MS']

        X, y = self.transform(I_in, I_PAN, I_MS)

        return X, y

    def __len__(self):
        
        return len(self.image_paths)