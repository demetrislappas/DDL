import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from copy import deepcopy

# Define the Score class which calculates anomaly scores for frames
class Score(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        # Rearrange the tensor to have channels last for visualization
        self.channel_last = Rearrange('b c h w -> b h w c')

    def numpy(self, x):
        # Convert a tensor to a numpy array
        return x.cpu().detach().numpy()

    def to_rgb(self, x):
        # If the input has a single channel, repeat it to create 3 channels (RGB)
        if x.shape[1] == 1:
            x = repeat(x, 'b c h w -> b (3 c) h w')
        return x

    def forward(self, x, x_hat):
        # If the input has a temporal dimension, select the target frame
        if len(x.shape) == 5:
            temporal = x.shape[2]
            target_frame = (temporal - 1) // 2
            x = x[:, :, target_frame]

        # Calculate the score map as the normalized difference between the original and reconstructed frames
        score_map = torch.norm(x - x_hat, dim=1) / (3 ** 0.5)
        
        half_patch_size = self.patch_size // 2
        # Exclude the border area of the score map to focus on the inner region
        score_map_inner = score_map[:, half_patch_size:-half_patch_size, half_patch_size:-half_patch_size]
        # Divide the score map into patches
        score_patches_outer = rearrange(score_map, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=self.patch_size, p2=self.patch_size)
        score_patches_inner = rearrange(score_map_inner, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=self.patch_size, p2=self.patch_size)
        # Concatenate the outer and inner patches
        score_patches = torch.cat([score_patches_outer, score_patches_inner], dim=1)
        # Calculate the mean score for each patch and find the maximum score across all patches
        score_number = score_patches.mean(dim=-1).max(dim=-1)[0]
        
        # Convert the original and reconstructed frames to RGB for visualization
        x = self.to_rgb(x)
        x_hat = self.to_rgb(x_hat)

        return [
            255 * self.numpy(self.channel_last(x)), 
            255 * self.numpy(self.channel_last(x_hat)), 
            self.numpy(score_map), 
            score_number.cpu().detach().numpy()
        ]
