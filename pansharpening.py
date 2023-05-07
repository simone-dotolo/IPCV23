import argparse
import torch
import numpy as np
from networks import APNN
from sensor import Sensor
from skimage.transform import rescale
from scipy.io import loadmat
from matplotlib import pyplot as plt


def main(args):

    model_name = args.model
    sensor_name = args.sensor
    input_path = args.input
    weights_path = args.weights
    use_gpu = args.use_gpu

    device = 'cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu'

    s = Sensor(sensor=sensor_name)

    if model_name == 'APNN':
        model = APNN(input_channels=s.nbands+1,
                     kernels=s.kernels).to(device)

    model.load_state_dict(torch.load(f=weights_path,
                                     map_location=torch.device(device)))
    
    I = loadmat(input_path)

    I_PAN, I_MS = I['I_PAN'], I['I_MS']

    # I_in concatenation of I_MS upsampled to I_PAN dimensions and I_PAN
    I_MS_UP = rescale(image=I_MS,
                      scale=[s.ratio, s.ratio, 1],
                      order=3)
    
    I_PAN = np.expand_dims(I_PAN, axis=-1)

    # Normalization
    I_in = np.concatenate([I_MS_UP, I_PAN], axis=-1) / (2 ** s.nbits)

    I_in = torch.from_numpy(I_in).permute([2,0,1]).unsqueeze(dim=0).to(device)

    model.eval()
    with torch.inference_mode():
        output = model(I_in)
    
    I_in = I_in.squeeze(dim=0).permute([1,2,0]).detach().cpu().numpy()
    output = output.squeeze(dim=0).permute([1,2,0]).detach().cpu().numpy()

    plt.figure(figsize=(15,6))

    plt.subplot(121)
    plt.imshow(I_in[:,:,(4,2,1)])

    plt.subplot(122)
    plt.imshow(output[:,:,(4,2,1)])

    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, help='Model to train', required=True, choices=['APNN'])
    parser.add_argument('-s', '--sensor', type=str, help='Sensor that acquired the image', required=True, choices=['QB', 'GE1', 'GeoEye1', 'WV2', 'WV3', 'Ikonos', 'IKONOS'])
    parser.add_argument('-i', '--input', type=str, help='Path to image that will be pansharpened', required=True)
    parser.add_argument('-w', '--weights', type=str, help='Path to models weights', required=True)
    parser.add_argument('--use_gpu', type=bool, action=argparse.BooleanOptionalAction, help='Enable GPU usage', required=False, default=False)

    args = parser.parse_args()

    main(args)