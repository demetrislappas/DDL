import argparse
from os.path import normpath
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from scripts.make_direcotries import md
from dataloaders.DatasetInfo import dataset_info
from dataloaders.VideoDataset import VideoDataset
from objects.models import Conv3dSkipUNet
from objects.pseudo_loss import PseudoLoss, PseudoAnomalyCreator

# The train function is defined to handle the training process
def train(args):
    # Hyperparameters are initialized based on the arguments passed
    methodology_name = 'DDL_V1'
    dataset_name = args.dataset_name
    training_batch_size = args.training_batch_size
    temporal = args.temporal
    resize = (args.resize_height, args.resize_width)
    epochs = args.epochs
    lr = args.lr
    device = args.device
    channels = args.channels

    # Get the training dataset path from the dataset info dictionary
    train_path = normpath(dataset_info[dataset_name]['train_path'])

    # Define the path for saving the trained model's features
    train_feature_filepath = f'./models/train_{methodology_name}_{dataset_name}/train_{methodology_name}_{dataset_name}.p'
    
    # Instantiate the model and the pseudo anomaly creator
    model = Conv3dSkipUNet(latent_dim=512, channels=channels, temporal=temporal).to(device=device)
    pseudo_anomaly_creator = PseudoAnomalyCreator().to(device)

    # Enable parallel training on multiple GPUs
    model = nn.DataParallel(model)
    pseudo_anomaly_creator = nn.DataParallel(pseudo_anomaly_creator)

    print(f'Training {methodology_name} for {dataset_name}')

    # Create the training dataset
    dataset_train = VideoDataset(directory=train_path, temporal=temporal, resize=resize)

    # Define the loss function specific to pseudo anomalies
    loss_func = PseudoLoss(temporal)

    # Define the optimizer and include both model and pseudo_anomaly_creator parameters
    parameters = list(model.parameters()) + list(pseudo_anomaly_creator.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)

    # Define a learning rate scheduler to adjust the learning rate during training
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * .8), gamma=.5, verbose=False)

    # Initialize a dictionary to store loss values for analysis
    loss_lists = {}

    # Training loop for each epoch
    for epoch in range(epochs):
        model.train()
        dataloader_train = DataLoader(dataset_train, batch_size=training_batch_size, shuffle=True)
        pbar = tqdm(dataloader_train, f'| training epoch {epoch+1}/{epochs} |')

        # Iterate over each batch in the training data
        for (x, _, mask) in pbar:
            if mask.sum() > 0:  # Check if there are any anomalies in the mask
                x = x.to(device)
                mask = mask.to(device)
                # Generate pseudo anomalies using the pseudo_anomaly_creator
                x_anom, anomaly_weight = pseudo_anomaly_creator(x, mask)

                # Get the model's reconstruction for normal and anomalous frames
                normal_reconstruction = model(x)
                anomalous_reconstruction = model(x_anom)

                # Calculate the loss
                loss = loss_func(x, x_anom, normal_reconstruction, anomalous_reconstruction, mask, anomaly_weight)
                optimizer.zero_grad()
                loss['loss'].backward()  # Backpropagate the loss
                optimizer.step()  # Update the model parameters

                # Log the loss details for progress display
                loss_details = []
                for loss_type in loss:
                    loss_details.append(f'{loss_type}: {np.round(float(loss[loss_type].mean()), 5)}')
                    if loss_type not in loss_lists:
                        loss_lists[loss_type] = []
                    loss_lists[loss_type].append(np.round(float(loss[loss_type].mean()), 3))
                pbar.set_description(f"| training epoch {epoch+1}/{epochs} | {', '.join(loss_details)} |")

        pbar.close()
        scheduler.step()  # Adjust the learning rate

    # Save the trained model's state
    torch.save(model.state_dict(), md(train_feature_filepath))
    print(f'| saved | Saved {methodology_name} in {train_feature_filepath}')

# Entry point for the script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training for DDL_V1')
    # Add command line arguments for the script
    parser.add_argument('--dataset_name', type=str, default='ped2', help='Name of the dataset')
    parser.add_argument('--training_batch_size', type=int, default=10, help='Training batch size')
    parser.add_argument('--temporal', type=int, default=3, help='Temporal length')
    parser.add_argument('--resize_height', type=int, default=256, help='Height to resize images')
    parser.add_argument('--resize_width', type=int, default=256, help='Width to resize images')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--channels', type=int, default=3, help='Number of channels in the input data')
    
    args = parser.parse_args()
    train(args)
