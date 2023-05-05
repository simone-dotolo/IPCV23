import os
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from scipy.io import loadmat

class PAN_Dataset(Dataset):
    
    def __init__(self, images_dir, sensor):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.sensor = sensor

    def transform(self, image, target):

        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x : x.type(torch.float32) / (2 ** self.sensor.nbits))
        ])

        return transf(image), transf(target)

    def __getitem__(self, index):
        
        image = loadmat(self.image_paths[index])

        X, y = image['I_in'], image['I_MS']

        X, y = self.transform(X, y)

        return X, y

    def __len__(self):
        
        return len(self.image_paths)