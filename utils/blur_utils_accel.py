import cv2
import numpy as np
from collections import deque
import argparse
import os
from moviepy import VideoFileClip
import torch

def apply_motion_trails(video_path, output_path, trail_indices, num_echos=7, decay_factor=0.65):
    os.makedirs(output_path, exist_ok=True)

    video = VideoFileClip(video_path)
    fps = video.fps
    if fps is None:
        print(f"Error: Could not determine FPS for {video_path}. Please provide it or ensure video has valid FPS.")
        video.close()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device}")

    def get_frame_from_moviepy(idx):
        time_in_seconds = idx / fps
        if time_in_seconds < 0 or time_in_seconds >= video.duration:
            # Frame index is out of bounds for the video
            return None

        # MoviePy's get_frame returns RGB, convert to BGR for OpenCV
        frame_rgb = video.get_frame(time_in_seconds)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return frame_bgr

    for cur_idx, start_idx in enumerate(trail_indices):
        # Calculate the end frame of the current trail sequence
        end_idx = start_idx + num_echos - 1

        # Check if the end_idx is within video bounds
        if end_idx >= video.duration * fps:
            last_frame_idx = int(video.duration * fps) - 1
            shift_amount = end_idx - last_frame_idx
            end_idx = last_frame_idx
            start_idx = start_idx - shift_amount
            # print(f"\033[93mWarning: Trail ending adjusted. New range: start={start_idx}, end={end_idx}\033[0m")

            if start_idx < 0:
                start_idx = 0
                print(f"\033[93mWarning: Start index adjusted to 0. Trail length reduced.\033[0m")

        # --- Step 1: Collect all 'num_echos' frames for the current trail sequence ---
        history = deque(maxlen=num_echos)
        all_frames_collected = True
        for i in range(num_echos):
            current_history_frame_idx = start_idx + i
            frame_to_add = get_frame_from_moviepy(current_history_frame_idx)

            if frame_to_add is None:
                print(f"Warning: Could not get frame {current_history_frame_idx} for trail starting at {start_idx}. Skipping this trail.")
                all_frames_collected = False
                break
            history.append(frame_to_add)

        if not all_frames_collected:
            continue # Move to the next trail_index

        # --- Step 2: Apply the blending effect using the collected history ---
        output_frame_bgr = history[-1].copy() # Deep copy to avoid modifying original

        accumulated_output = torch.from_numpy(history[-1].copy()).to(device).float()
        current_decay = decay_factor # This decay applies to the *next* older frame

        # Iterate from the second-to-last frame down to the oldest (history[0])
        for i in range(len(history) - 2, -1, -1):
            past_frame = history[i] # This is an older frame
            past_frame_tensor = torch.from_numpy(past_frame).to(device).float()

            accumulated_output = past_frame_tensor * current_decay + accumulated_output * (1 - current_decay)
            current_decay *= decay_factor

        output_frame_bgr = accumulated_output.cpu().numpy().astype(np.uint8)

        # --- Step 3: Save the processed frame ---
        output_frame_path = os.path.join(output_path, f"{cur_idx:04d}.jpg")
        cv2.imwrite(output_frame_path, output_frame_bgr)

    video.close()



def apply_background_blur(frames_path, video_segments, output_dir, blur_radius=99):
    """
    Apply blur effect to the background of video frames based on video_segments.

    Args:
        frames_path (str): Path to the directory containing input frames
        video_segments (dict): Dictionary containing frame_idx and id2masks
        output_dir (str): Directory to save the blurred frames
        blur_radius (int): Radius of the Gaussian blur kernel
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each frame
    for frame_idx, id2masks in video_segments.items():
        # Read the original frame
        frame_path = os.path.join(frames_path, f"{frame_idx:04d}.jpg")
        if not os.path.exists(frame_path):
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Create a mask for the background (inverse of all object masks)
        height, width = frame.shape[:2]
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        # Combine all object masks
        for obj_id, mask in id2masks.items():
            # Convert mask to uint8 if it's boolean
            if mask.dtype == bool:
                mask = mask.astype(np.uint8) * 255

            # Ensure mask is 2D by taking the first channel if needed
            if len(mask.shape) == 3:
                mask = mask[0]

            # Add this object's mask to the combined mask
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Invert the combined mask to get background mask
        background_mask = cv2.bitwise_not(combined_mask)

        # Apply extreme Gaussian blur to the entire frame
        kernel_size = (blur_radius, blur_radius)
        sigma = blur_radius/3
        blurred_frame = cv2.GaussianBlur(frame, kernel_size, sigma)

        blurred_frame = cv2.GaussianBlur(blurred_frame, kernel_size, sigma)

        # Create output frame by combining original and blurred regions
        output_frame = frame.copy()
        # Where background_mask is 255 (background), use blurred frame
        output_frame[background_mask == 255] = blurred_frame[background_mask == 255]

        # Save the processed frame
        output_path = os.path.join(output_dir, f"{frame_idx:04d}.jpg")
        cv2.imwrite(output_path, output_frame)

    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply motion trail effect to video')
    parser.add_argument('--input', type=str, required=True, help='Input video file path')
    parser.add_argument('--output', type=str, required=True, help='Output video file path')
    parser.add_argument('--num_echos', type=int, default=7, help='Number of trail frames (default: 7)')
    parser.add_argument('--decay_factor', type=float, default=0.65, help='Trail decay factor (0.0-1.0, default: 0.65)')

    args = parser.parse_args()
    apply_motion_trails(args.input, args.output, args.num_echos, args.decay_factor)