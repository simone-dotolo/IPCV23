import argparse

import torch
from tqdm.auto import tqdm

from data import PAN_Dataset
from metrics import SAM, ERGAS, Q, Q2n, ReproMetrics, DRho
from networks import APNN, DRPNN
from sensor import Sensor

def eval(args):
    use_gpu = args.use_gpu
    device = 'cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu'

    model_name = args.model
    weights_path = args.weights
    test_path = args.test_fold 
    sensor_name = args.sensor
    full_resolution = args.full_resolution

    s = Sensor(sensor=sensor_name)

    test_dataset = PAN_Dataset(images_dir=test_path,
                               sensor=s,
                               full_resolution=full_resolution,
                               eval=True)

    model_name = args.model

    if model_name == 'APNN':
        model = APNN(input_channels=s.nbands+1,
                     kernels=s.kernels).to(device)
    elif model_name == 'DRPNN':
        model = DRPNN(input_channels=s.nbands+1).to(device)
    
    model.load_state_dict(torch.load(f=weights_path,
                                     map_location=torch.device(device)))

    SAM_value = 0
    ERGAS_value = 0
    Q_value = 0
    Q2n_value = 0
    DRho_value = 0

    model.eval()
    with torch.inference_mode():
        
        if full_resolution:
            for X, I_MS, I_PAN in tqdm(test_dataset):
                
                X = X.unsqueeze(dim=0).to(device)

                output = model(X)

                # Moving tensors to cpu and converting to numpy arrays
                output = torch.permute(output.squeeze(dim=0), [2, 1, 0]).detach().cpu().numpy()
                I_MS = torch.permute(I_MS, [2, 1, 0]).detach().cpu().numpy()
                I_PAN = I_PAN.squeeze(dim=0).detach().cpu().numpy()

                q2n, q, sam, ergas = ReproMetrics(output, I_MS, I_PAN, s.sensor, s.ratio)
                drho = DRho(output, I_PAN, sigma=s.ratio)

                SAM_value += sam
                ERGAS_value += ergas
                Q_value += q
                Q2n_value += q2n         
                DRho_value += drho

            SAM_value /= len(test_dataset)
            ERGAS_value /= len(test_dataset)
            Q_value /= len(test_dataset)
            Q2n_value /= len(test_dataset)
            DRho_value /= len(test_dataset)

            print(f'FULL RESOLUTION!\nSAM: {SAM_value} | ERGAS: {ERGAS_value} | Q: {Q_value} | Q2n: {Q2n_value} | DRho: {DRho_value}')
        else:
            for X, y in tqdm(test_dataset):
                
                X = X.unsqueeze(dim=0).to(device)
                output = model(X)

                output = torch.permute(output.squeeze(dim=0), [2, 1, 0]).detach().cpu().numpy()
                y = torch.permute(y, [2, 1, 0]).detach().cpu().numpy()

                SAM_value += SAM(output, y)
                ERGAS_value += ERGAS(output, y, s.ratio)
                Q_value += Q(output, y)
                Q2n_value += Q2n(output, y)[0]        

            SAM_value /= len(test_dataset)
            ERGAS_value /= len(test_dataset)
            Q_value /= len(test_dataset)
            Q2n_value /= len(test_dataset)
            print(f'REDUCED RESOLUTION!\nSAM: {SAM_value} | ERGAS: {ERGAS_value} | Q: {Q_value} | Q2n: {Q2n_value}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, help='Model to train', required=True, choices=['APNN', 'DRPNN'])
    parser.add_argument('-w', '--weights', type=str, help='Path to model weights', required=True)
    parser.add_argument('-s', '--sensor', type=str, help='Sensor that acquired the image', required=True, choices=['QB', 'GE1', 'GeoEye1', 'WV2', 'WV3', 'Ikonos', 'IKONOS'])
    parser.add_argument('-t', '--test_fold', type=str, help='Path to training set', required=True)
    parser.add_argument('--use_gpu', type=bool, action=argparse.BooleanOptionalAction, help='Enable GPU usage', required=False, default=False)
    parser.add_argument('--full_resolution', type=bool, action=argparse.BooleanOptionalAction, help='Working in the FULL RESOLUTION Framework', required=False, default=False)
    
    args = parser.parse_args()

    eval(args)