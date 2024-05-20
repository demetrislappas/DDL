# Import native packages
import os
import json

# Import external packages
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
from natsort import natsorted

class VideoDataset(Dataset):
    def __init__(self, directory, temporal=3, resize=None, scene=None):
        # Initialize the dataset with the directory, temporal length, resize dimensions, and scene filter
        self.directory = directory
        self.temporal = temporal
        self.orig_size = None
        
        # Set up the resizing transform if specified
        self.resize = torchvision.transforms.Resize(resize) if resize is not None else None

        # Get a list of all subdirectories in the dataset directory
        subdirectories = [os.path.join(self.directory, d) for d in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory, d))]
        subdirectories = natsorted(subdirectories)
        # Filter subdirectories based on the scene if specified
        subdirectories = [subdirectory for subdirectory in subdirectories if subdirectory.split('/')[-1].split('_')[0] == scene] if scene is not None else subdirectories
        
        # Initialize lists to hold frame file paths and their corresponding indices
        self.frame_files = []
        self.frame_files_index = []
        self.mask_files = {}
        self.temp_scale = (temporal - 1) // 2

        # Populate the frame file lists with image files from the subdirectories
        for subdirectory in subdirectories:
            subdirectory_files = [os.path.join(subdirectory, f) for f in os.listdir(subdirectory) if os.path.isfile(os.path.join(subdirectory, f)) and (f[-3:] in ['png', 'jpg', 'tif'])]
            subdirectory_files = sorted(subdirectory_files)
            self.frame_files += subdirectory_files
            self.frame_files_index += subdirectory_files[self.temp_scale:-self.temp_scale] if temporal > 1 else subdirectory_files
        
        # Load bounding box annotations if available
        bboxes_path = f'{self.directory}/tracked_bounding_boxes.json'
        self.bounding_boxes = json.load(open(bboxes_path, 'r')) if os.path.isfile(bboxes_path) else None

    def __len__(self):
        # Return the length of the dataset
        return len(self.frame_files_index)

    def get_common_tracks(self, frames):
        # Get common tracked objects across the given frames
        track_lists = []
        for frame in frames:
            tracks = [track for track in self.bounding_boxes[frame]] if frame in self.bounding_boxes else []
            track_lists.append(tracks)

        # Find common tracks across all frames
        common_tracks = set(track_lists[0])
        for track_list in track_lists[1:]:
            common_tracks = common_tracks & set(track_list)

        return list(common_tracks)

    def __getitem__(self, idx, fixed_track=None):
        # Get the file name of the frame at the given index
        filename = self.frame_files_index[idx]
        temp_idx = self.frame_files.index(filename)
        start_frame = temp_idx - self.temp_scale
        end_frame = temp_idx + self.temp_scale

        # Get the file names of the frames in the temporal window
        frame_files = self.frame_files[start_frame:end_frame+1]

        # Load and stack the frames into a tensor
        frames = [torchvision.io.read_image(frame_file) for frame_file in frame_files]
        frames = torch.stack(frames, dim=1)

        # Normalize the frame pixel values
        frames = frames / 255
        if self.resize is not None:
            frames = self.resize(frames)

        # If bounding boxes are available and common tracks are found
        if self.bounding_boxes is not None and len(self.get_common_tracks(frame_files)) > 0:
            common_tracks = self.get_common_tracks(frame_files)
            chosen_track = common_tracks[np.random.randint(len(common_tracks))] if fixed_track is None else common_tracks[fixed_track]
            height, width = frames.shape[-2:]
            box_scale = np.array([width, height, width, height])
            # Get bounding boxes for the chosen track in each frame
            boxes = [self.bounding_boxes[frame_file][chosen_track]['box'] * box_scale for frame_file in frame_files]
        else:
            boxes = []

        # Create masks for the bounding boxes
        masks = torch.zeros_like(frames) 
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box.astype(np.int32)
            masks[:, i, y_min:y_max, x_min:x_max] = 1

        # Return the frames, file name, and masks
        return (frames, filename, masks)
