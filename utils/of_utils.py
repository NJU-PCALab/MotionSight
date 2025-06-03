import os
import torch
import argparse
from decord import VideoReader, cpu
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from RAFT.core.raft import RAFT
from RAFT.core.utils import flow_viz

raft_args = argparse.Namespace(
    alternate_corr=False,
    model="./RAFT/models/raft-things.pth",
    small=False,
    mixed_precision=False,
)
raft = torch.nn.DataParallel(RAFT(raft_args))
raft.load_state_dict(torch.load(raft_args.model))
raft = raft.module.cuda()
raft.eval()

def extract_of(frames_path, len_frames=None):
    """
    Extract optical flow features from frames

    Args:
        frames_path: path to the frames directory
        len_frames: optional limit on number of frames to process

    Returns:
        flow_features: (T-1, 2, H, W) or empty tensor if only one frame
    """
    full_frames_path = sorted([os.path.join(frames_path, f) for f in os.listdir(frames_path) if f.endswith('.png') or f.endswith('.jpg')])
    if len_frames is not None:
        full_frames_path = full_frames_path[:len_frames]
    video = [Image.open(frame_path).convert('RGB') for frame_path in full_frames_path]

    # Handle case with only one frame
    if len(video) <= 1:
        return torch.zeros((0, 2, 0, 0))

    # Transform to convert PIL images to tensors
    to_tensor = transforms.ToTensor()

    flows = []
    for t in range(len(video)-1):
        # Resize images to ensure dimensions are divisible by 8
        img1 = video[t]
        img2 = video[t+1]

        # Get original dimensions
        orig_width, orig_height = img1.size

        # Calculate new dimensions divisible by 8
        new_width = ((orig_width + 7) // 8) * 8
        new_height = ((orig_height + 7) // 8) * 8

        # Resize images
        img1_resized = img1.resize((new_width, new_height), Image.BILINEAR)
        img2_resized = img2.resize((new_width, new_height), Image.BILINEAR)

        # Convert PIL images to tensors [C, H, W] and add batch dimension
        frame1 = to_tensor(img1_resized).unsqueeze(0).cuda()
        frame2 = to_tensor(img2_resized).unsqueeze(0).cuda()

        # Ensure pixel values are in range [0, 255]
        frame1 = frame1 * 255.0
        frame2 = frame2 * 255.0

        with torch.no_grad():
            flow_low, flow_up = raft(frame1, frame2, iters=20, test_mode=True)

            # Resize flow back to original dimensions if needed
            if new_width != orig_width or new_height != orig_height:
                # Resize flow field to original dimensions
                flow_up = torch.nn.functional.interpolate(
                    flow_up, size=(orig_height, orig_width), mode='bilinear', align_corners=False
                )
                # Scale flow values to account for dimension changes
                flow_up[:, 0] = flow_up[:, 0] * (orig_width / new_width)
                flow_up[:, 1] = flow_up[:, 1] * (orig_height / new_height)

            flows.append(flow_up)

    flows = torch.cat(flows, dim=0)  # [T-1, 2, H, W]

    return flows

def visualize_flow(flows, save_dir):
    save_dir = f"{save_dir}"
    import os
    os.makedirs(save_dir, exist_ok=True)

    if torch.is_tensor(flows):
        flows = flows.cpu().numpy()

    for t in range(len(flows)):
        flow = flows[t]  # [2, H, W]
        flow = flow.transpose(1, 2, 0)  # [H, W, 2]
        flow_rgb = flow_viz.flow_to_image(flow)
        cv2.imwrite(f'{save_dir}/flow_{t:03d}.jpg', flow_rgb[:, :, [2,1,0]])


# file = "/home/myname/motion-sight/pexels_landscape_landscape_10203249_002.mp4"
