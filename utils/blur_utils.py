import cv2
import numpy as np
from collections import deque
import argparse
import os

def apply_motion_trails(video_path, output_path, num_echos=7, decay_factor=0.65):
    """
    Apply motion trail effect to video without mask.

    Args:
        video_path (str): Input video file path
        output_path (str): Output video file path
        num_echos (int): Number of trail frames
        decay_factor (float): Trail decay factor (0.0-1.0)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Error: Cannot create output video file {output_path}")
        cap.release()
        return

    history = deque(maxlen=num_echos)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = frame.copy()
        history.append(frame.copy())

        current_decay = decay_factor
        for i in range(len(history) - 1, -1, -1):
            past_frame = history[i]
            alpha = current_decay
            output_frame = cv2.addWeighted(past_frame, alpha, output_frame, 1 - alpha, 0)
            current_decay *= decay_factor

        out.write(output_frame)

    cap.release()
    out.release()
    # cv2.destroyAllWindows()


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