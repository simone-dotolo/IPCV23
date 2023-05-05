import torch
from torch import nn
from torch.utils.data import DataLoader
from data import PAN_Dataset
from sensor import Sensor
from networks import APNN
from tqdm.auto import tqdm
import argparse

def train(args):

    # Device agnostic code
    use_gpu = args.use_gpu
    device = 'cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu'
    
    sensor_name = args.sensor
    s = Sensor(sensor_name)

    # Hyperparameters
    epochs = args.epochs
    batch_size = args.batch
    learning_rate = args.lr
    optim_name = args.optim
    loss_name = args.loss

    # Data folders
    train_path = args.train_fold
    valid_path = args.val_fold

    # Getting datasets
    train_dataset = PAN_Dataset(images_dir=train_path,
                                sensor=s)
    valid_dataset = PAN_Dataset(images_dir=valid_path,
                                sensor=s)

    # Creating dataloaders
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    valid_dataloader = DataLoader(dataset=valid_dataset, 
                                  batch_size=batch_size,
                                  shuffle=False)

    # Getting a model
    model_name = args.model

    if model_name == 'APNN':
        model = APNN(input_channels=s.nbands+1,
                     kernels=s.kernels).to(device)

    # Loss
    if loss_name == 'L1':
        loss_fn = nn.L1Loss()
    elif loss_name == 'L2':
        loss_fn = nn.MSELoss()

    # Optimizer
    if optim_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate)
    elif optim_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters,
                                    lr=learning_rate) 

    # Training
    train_losses = []
    valid_losses = []

    for epoch in tqdm(range(epochs)):

        train_loss = 0

        model.train()

        for (X, y) in train_dataloader:

            X, y = X.to(device), y.to(device)

            output = model(X)

            loss = loss_fn(output, y)
            train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
        
        valid_loss = 0

        model.eval()
        with torch.inference_mode():

            for (X, y) in valid_dataloader:

                X, y = X.to(device), y.to(device)

                output = model(X)

                loss = loss_fn(output, y)
                valid_loss += loss.item()
        
        train_loss /= len(train_dataloader)
        valid_loss /= len(valid_dataloader)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(f'Epoch: {epoch} | Train loss: {train_loss:.4f} | Valid loss: {valid_loss:.4f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='Model to train', required=True, choices=['APNN'])
    parser.add_argument('-s', '--sensor', type=str, help='Sensor that acquired the image', required=True, choices=['QB', 'GE1', 'GeoEye1', 'WV2', 'WV3', 'Ikonos', 'IKONOS'])
    parser.add_argument('-t', '--train_fold', type=str, help='Path to training set', required=True)
    parser.add_argument('-v', '--val_fold', type=str, help='Path to validation set', required=True)
    parser.add_argument('--use_gpu', type=bool, help='Enable GPU usage', required=False, default=False)
    # Hyperparams
    parser.add_argument('--batch', type=int, help='Batch size', required=False, default=16)
    parser.add_argument('--lr', type=float, help='Initial learning rate', required=False, default=1e-3)
    parser.add_argument('--epochs', type=int, help='Number of epochs', required=False, default=10)
    parser.add_argument('--optim', help='Optimizer used for the training phase', required=False, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--loss', help='Loss used for the training phase', required=False, default='L1', choices=['L1', 'L2'])

    args = parser.parse_args()

    train(args)