import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import PAN_Dataset
from early_stopper import EarlyStopper 
from losses import SpectralStructuralLoss
from networks import APNN, DRPNN
from sensor import Sensor

def train(args):
    use_gpu = args.use_gpu
    device = 'cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu'
    
    sensor_name = args.sensor
    s = Sensor(sensor=sensor_name)

    full_resolution = args.full_resolution

    # Hyperparameters
    epochs = args.epochs
    batch_size = args.batch
    learning_rate = args.lr
    lr_scheduler = args.lr_sched
    stop_early = args.early_stopper
    optim_name = args.optim
    loss_name = args.loss

    # Data folders
    train_path = args.train_fold
    valid_path = args.val_fold

    # Output folder
    output_folder = args.out_fold

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Getting datasets
    train_dataset = PAN_Dataset(images_dir=train_path,
                                sensor=s,
                                full_resolution=full_resolution)
    valid_dataset = PAN_Dataset(images_dir=valid_path,
                                sensor=s,
                                full_resolution=full_resolution)

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
    elif model_name == 'DRPNN':
        model = DRPNN(input_channels=s.nbands+1).to(device)

    # Loss
    if full_resolution:
        I_PAN_shape = tuple(train_dataset[0][0][0].shape)   
        print('Working in FULL RESOLUTION. Ignoring loss args...\n')
        loss_fn = SpectralStructuralLoss(img_pan_shape=I_PAN_shape,
                                         device=device,
                                         sensor=s).to(device)
    elif loss_name == 'L1':
        loss_fn = nn.L1Loss()
    elif loss_name == 'L2':
        loss_fn = nn.MSELoss()

    # Optimizer
    if optim_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate)
    elif optim_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    momentum=0.9) 
    
    # Learning rate scheduler
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               factor=0.1,
                                                               patience=10,
                                                               min_lr=1e-7)

    # Early stopping
    if stop_early:
        early_stopper = EarlyStopper(patience=20,
                                     min_delta=0)

    # Training
    train_losses = []
    valid_losses = []

    min_valid_loss = np.inf

    for epoch in tqdm(range(epochs)):

        train_loss = 0

        model.train()

        for (X, y) in train_dataloader:

            X, y = X.to(device), y.to(device)

            output = model(X)

            if full_resolution:
                loss = loss_fn(output, y, X)
            else:
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

                if full_resolution:
                    loss = loss_fn(output, y, X)
                else:
                    loss = loss_fn(output, y)
                valid_loss += loss.item()

        if lr_scheduler:
            scheduler.step(loss)
        
        train_loss /= len(train_dataloader)
        valid_loss /= len(valid_dataloader)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'Epoch: {epoch} | Train loss: {train_loss:.4f} | Valid loss: {valid_loss:.4f}')

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            model_path = model_name + '_weights_' + 'epoch_' + str(epoch) + '.pth'
            output_path = os.path.join(output_folder, model_path)
            torch.save(model.state_dict(), output_path)

        if stop_early and early_stopper.early_stopping(validation_loss=valid_loss):
            print(f'{early_stopper.patience} epochs without improvement. Early stopping...\n')
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='Model to train', required=True, choices=['APNN', 'DRPNN'])
    parser.add_argument('-s', '--sensor', type=str, help='Sensor that acquired the image', required=True, choices=['QB', 'GE1', 'GeoEye1', 'WV2', 'WV3', 'Ikonos', 'IKONOS'])
    parser.add_argument('-t', '--train_fold', type=str, help='Path to training set', required=True)
    parser.add_argument('-v', '--val_fold', type=str, help='Path to validation set', required=True)
    parser.add_argument('-o', '--out_fold', type=str, help='Path to output folder', required=True)
    parser.add_argument('--use_gpu', type=bool, action=argparse.BooleanOptionalAction, help='Enable GPU usage', required=False, default=False)
    parser.add_argument('--full_resolution', type=bool, action=argparse.BooleanOptionalAction, help='Working in the FULL RESOLUTION Framework', required=False, default=False)

    # Hyperparams
    parser.add_argument('--batch', type=int, help='Batch size', required=False, default=16)
    parser.add_argument('--lr', type=float, help='Initial learning rate', required=False, default=1e-3)
    parser.add_argument('--lr_sched', type=bool, action=argparse.BooleanOptionalAction, help='Enable learning rate scheduler', required=False, default=False)
    parser.add_argument('--early_stopper', type=bool, action=argparse.BooleanOptionalAction, help='Enable early stopping', required=False, default=False)
    parser.add_argument('--epochs', type=int, help='Number of epochs', required=False, default=1)
    parser.add_argument('--optim', type=str, help='Optimizer used for the training phase', required=False, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--loss', type=str, help='Loss used for the training phase', required=False, default='L1', choices=['L1', 'L2'])

    args = parser.parse_args()

    train(args)