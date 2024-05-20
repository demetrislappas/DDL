import argparse
from os.path import normpath
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

from scripts.make_direcotries import md
from dataloaders.DatasetInfo import dataset_info
from dataloaders.VideoDataset import VideoDataset
from objects.models import Conv3dSkipUNet
from objects.scoring import Score
from objects.video_writer import VideoWriter

# The test function is defined to handle the testing process
def test(args):
    # Hyperparameters are initialized based on the arguments passed
    methodology_name = 'DDL_V1'
    dataset_name = args.dataset_name
    testing_batch_size = 1
    temporal = args.temporal
    resize = (args.resize_height, args.resize_width)
    device = args.device
    channels = args.channels

    # Get the testing dataset path from the dataset info dictionary
    test_path = normpath(dataset_info[dataset_name]['test_path'])

    # Define the path for the trained model's features
    train_feature_filepath = f'./models/train_{methodology_name}_{dataset_name}/train_{methodology_name}_{dataset_name}.p'
    
    # Instantiate the model
    model = Conv3dSkipUNet(latent_dim=512, channels=channels, temporal=temporal).to(device=device)

    # Enable parallel testing on multiple GPUs
    model = nn.DataParallel(model)

    print(f'Testing {methodology_name} for {dataset_name}')

    # Load the trained model
    model.load_state_dict(torch.load(md(train_feature_filepath)))
    model.eval()

    # Create the testing dataset
    dataset_test = VideoDataset(directory=test_path, temporal=temporal, resize=resize)
    dataloader_test = DataLoader(dataset_test, batch_size=testing_batch_size)

    # Create a directory for saving visual results
    md(normpath(f'./visual/{dataset_name}/{methodology_name}'))
    vidgen = VideoWriter(f'./visual/{dataset_name}/{methodology_name}')

    # Initialize the scoring system
    score = Score()
    scores_dict = {}

    # Testing loop for each batch in the test data
    pbar = tqdm(dataloader_test, f'| testing | {dataset_name} |')
    for x, image_filename, _ in pbar:
        with torch.no_grad():  # Disable gradient calculation for testing
            x = x.to(device)
            x_hat = model(x)  # Get the model's reconstruction
            score_maps = score(x, x_hat)  # Calculate score maps for anomaly detection

        # Extract the video name from the image filename
        vid_name = image_filename[0].split('/')[-2]
        if vid_name not in scores_dict:
            scores_dict[vid_name] = []
        scores_dict[vid_name].append(score_maps[-1])
        vidgen.update(image_filename[0], score_maps)  # Update the video writer with the score maps

    pbar.close()
    vidgen.release()  # Release the video writer

    # Save the scores to a pickle file
    with open(md(normpath(f'./visual/{dataset_name}/{methodology_name}/{methodology_name}_scores.pkl')), 'wb') as f:
        pickle.dump(scores_dict, f)

# Entry point for the script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run testing for DDL_V1')
    # Add command line arguments for the script
    parser.add_argument('--dataset_name', type=str, default='ped2', help='Name of the dataset')
    parser.add_argument('--temporal', type=int, default=3, help='Temporal length')
    parser.add_argument('--resize_height', type=int, default=256, help='Height to resize images')
    parser.add_argument('--resize_width', type=int, default=256, help='Width to resize images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for testing')
    parser.add_argument('--channels', type=int, default=3, help='Number of channels in the input data')
    
    args = parser.parse_args()
    test(args)
