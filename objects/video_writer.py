import cv2
import os
import numpy as np
from einops import repeat

# Define the VideoWriter class which handles writing frames to video files
class VideoWriter:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.current_video_writer = None
        self.current_subfolder = None
        self.frame_size = None
        self.fps = 15  # Frames per second for the output video
        
    def _start_new_video(self, subfolder):
        # Release the current video writer if one is already active
        if self.current_video_writer is not None:
            self.current_video_writer.release()

        # Define the path and codec for the new video
        video_file_name = f"{subfolder}.avi"
        video_path = os.path.join(self.base_dir, video_file_name)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.current_video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, self.frame_size)
        self.current_subfolder = subfolder

    def update(self, frame_path, score_maps):
        # Read the current frame from the provided path
        frame = cv2.imread(frame_path)
        if self.frame_size is None:
            # Set the frame size for the video writer if not already set
            self.frame_size = (2 * frame.shape[1], 2 * frame.shape[0])

        # Determine the subfolder (video sequence) from the frame path
        subfolder = os.path.basename(os.path.dirname(frame_path))
        if subfolder != self.current_subfolder:
            # Start a new video if the subfolder has changed
            self._start_new_video(subfolder)

        # Extract score maps: resized frame, reconstructed frame, difference map, and anomaly score
        resized_frame, reconstructed_frame, difference_map, anomaly_score = score_maps

        # Resize frames to match the original frame size
        single_frame_size = (frame.shape[1], frame.shape[0])
        resized_frame = cv2.resize(resized_frame[0], single_frame_size).astype(np.uint8)[:, :, ::-1]
        reconstructed_frame = cv2.resize(reconstructed_frame[0], single_frame_size).astype(np.uint8)[:, :, ::-1]
        difference_map = cv2.resize(difference_map[0], single_frame_size)
        
        # Normalize the difference map to the range [0, 255] and repeat it across 3 channels for RGB
        difference_map = 255 * (difference_map - difference_map.min()) / (difference_map.max() - difference_map.min())
        difference_map = repeat(difference_map, 'h w -> h w 3').astype(np.uint8)
                
        # Concatenate the original frame and difference map horizontally
        top = np.concatenate([frame, difference_map], axis=-2)
        # Concatenate the resized and reconstructed frames horizontally
        bottom = np.concatenate([resized_frame, reconstructed_frame], axis=-2)
        # Concatenate the top and bottom parts vertically to form the final frame
        frame = np.concatenate([top, bottom], axis=0)
       
        # Write the final frame to the current video
        self.current_video_writer.write(frame)

    def release(self):
        # Release the current video writer
        if self.current_video_writer is not None:
            self.current_video_writer.release()
