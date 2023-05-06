import torch
from torch.utils.data import DataLoader
from sensor import Sensor
from networks import APNN
from data import PAN_Dataset
from metrics import SAM, ERGAS, Q, Q2n
import argparse
from tqdm.auto import tqdm

def eval(args):

    # Device agnostic code
    use_gpu = args.use_gpu
    device = 'cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu'

    print(device)

    model_name = args.model
    weights_path = args.weights
    test_path = args.test_fold 
    sensor_name = args.sensor

    s = Sensor(sensor=sensor_name)

    test_dataset = PAN_Dataset(images_dir=test_path,
                               sensor=s)

    model_name = args.model

    if model_name == 'APNN':
        model = APNN(input_channels=s.nbands+1,
                     kernels=s.kernels).to(device)
    
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))

    SAM_value = 0
    ERGAS_value = 0
    Q_value = 0
    Q2n_value = 0

    model.eval()
    with torch.inference_mode():
        
        for X, y in tqdm(test_dataset):
            
            X, y = X.unsqueeze(dim=0).to(device), y.unsqueeze(dim=0).to(device)

            output = model(X)

            output = torch.permute(output.squeeze(dim=0), [2, 1, 0]).detach().cpu().numpy()
            y = torch.permute(y.squeeze(dim=0), [2, 1, 0]).detach().cpu().numpy()

            SAM_value += SAM(output, y)
            ERGAS_value += ERGAS(output, y, s.ratio)
            Q_value += Q(output, y)
            Q2n_value += Q2n(output, y)[0]        

    SAM_value /= len(test_dataset)
    ERGAS_value /= len(test_dataset)
    Q_value /= len(test_dataset)
    Q2n_value /= len(test_dataset)

    print(f'SAM: {SAM_value} | ERGAS: {ERGAS_value} | Q: {Q_value} | Q2n: {Q2n_value}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, help='Model to train', required=True, choices=['APNN'])
    parser.add_argument('--weights', type=str, help='Path to model weights', required=True)
    parser.add_argument('-s', '--sensor', type=str, help='Sensor that acquired the image', required=True, choices=['QB', 'GE1', 'GeoEye1', 'WV2', 'WV3', 'Ikonos', 'IKONOS'])
    parser.add_argument('-t', '--test_fold', type=str, help='Path to training set', required=True)
    parser.add_argument('--use_gpu', type=bool, action=argparse.BooleanOptionalAction, help='Enable GPU usage', required=False, default=False)

    args = parser.parse_args()

    eval(args)