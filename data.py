import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from scipy.io import loadmat

class PAN_Dataset(Dataset):
    
    def __init__(self, images_dir):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
    
    def transform(self, image, target):

        return transforms.ToTensor(image), transforms.ToTensor(target)

    def __getitem__(self, index):
        
        image = loadmat(self.image_paths[index])

        I_in, I_MS = image['I_in'], image['I_MS']

        X = Image.open(I_in)
        y = Image.open(I_MS) 

        X, y = self.transform(X, y)

        return X, y

    def __len__(self):
        
        return len(self.image_paths)

mat = loadmat('Adelaide_1.mat')

I_in = mat['I_in']
I_MS = mat['I_MS']
I_MS_LR = mat['I_MS_LR']
I_PAN = mat['I_PAN']
I_PAN_LR = mat['I_PAN_LR']

print(f'I_in shape: {I_in.shape}')
print(f'I_MS shape: {I_MS.shape}')
print(f'I_MS_LR shape: {I_MS_LR.shape}')
print(f'I_PAN shape: {I_PAN.shape}')
print(f'I_PAN_LR shape: {I_PAN_LR.shape}')

plt.figure()

plt.subplot(121)
plt.imshow(I_PAN, cmap='gray')
plt.title('I_PAN')

plt.subplot(122)
plt.imshow(I_MS[:,:,(4,2,1)] / 2048.0)
plt.title('I_MS')

plt.show()