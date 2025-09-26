import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Give directory of checkpoint')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--motionbench_pos', type=str,
                        help='Path to MotionBench')
    parser.add_argument('--exp_name', type=str,
                        default="motionsight",
                        help='Path to MotionBench')
    return parser.parse_args()

args = parse_args()
global_name = args.exp_name
checkpoint_path = args.checkpoint_path

## global parameters
do_fps = True # You may need change this according to model
num_segs = 16
resize_for_memory = True
new_rate = 1
num_segs_dense = 16

import numpy as np
import json
from PIL import Image, ImageDraw
import os
import time
import torch
import random
from tqdm import tqdm
import sys
from io import StringIO
from utils.blur_utils import apply_motion_trails, apply_background_blur
from multiprocessing import shared_memory
from datetime import datetime
from lmdeploy.vl.constants import IMAGE_TOKEN
from utils.utils import *
from functools import wraps
import shutil
from typing import List, Dict, Any
from moviepy import VideoFileClip
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy import pipeline, TurbomindEngineConfig, PytorchEngineConfig, GenerationConfig, ChatTemplateConfig
import torch.distributed as dist

output_filename = f'{global_name}.jsonl'
restore_name = f'restore_{global_name}'

# prompt template
class VideoConfig:
    def __init__(self, path=None, num_frames=0, start_frame=0, rate=1):
        self.path = path
        self.num_frames = num_frames
        self.start_frame = start_frame
        self.rate = rate

    def is_valid(self):
        return self.path is not None and self.num_frames > 0

def ask_mllm(pipe, q, video_path=None, history=None, len_frames=None, debug=True, start_frame=0, rate=1):
    if video_path is not None:
        question = ''
        for i in range(len_frames):
            # Calculate frame number with proper formatting
            frame_num = i/rate+start_frame
            if frame_num == int(frame_num):
                question = question + f'Frame{int(frame_num)}: {IMAGE_TOKEN}\n'
            else:
                question = question + f'Frame{frame_num:.2f}: {IMAGE_TOKEN}\n'
        question += q
    else:
        question = q

    content = [{'type': 'text', 'text': question}]

    if video_path is not None:
        for i in range(len_frames):
            image_url_dict = {
                'max_dynamic_patch': 1,
                'url': os.path.abspath(f'{video_path}/{i:04d}.jpg')
            }

            if resize_for_memory:
                image_url_dict['max_pixels'] = 360 * 420

            content.append({
                'type': 'image_url',
                'image_url': image_url_dict
            })

    if history is None:
        history = [dict(role='user', content=content)]
        response = pipe(history, gen_config=GenerationConfig(max_new_tokens=128, do_sample=True))
        output_text = response.text
        history.append(dict(role='assistant', content=output_text))
    else:
        history.append(dict(role='user', content=content))
        response = pipe(history, gen_config=GenerationConfig(max_new_tokens=128, do_sample=True))
        output_text = response.text
        history.append(dict(role='assistant', content=output_text))

    print(history)
    return output_text, history

def ask_mllm_multi(pipe, q, video_configs=None, history=None, debug=True):
    """
    Ask MLLM with multiple video inputs

    Args:
        q (str): Question text
        video_configs (dict): Dictionary of video configurations, keys are video types, values are VideoConfig objects
                             Supported keys: 'original', 'spotlight', 'motion_blur', 'background_blur'
        history (list): Conversation history
        do_log (bool): Whether to log the conversation
        debug (bool): Whether to output debug information

    Returns:
        tuple: (response content, updated history)
    """
    if video_configs is None:
        video_configs = {}

    question = ''

    # Build prompts containing various video frames
    video_descriptions = {
        'original': 'Original video:\n',
        'spotlight': 'Spotlight video:\n',
        'motion_blur': 'Original video with motion blur to more clearly determine the type of motion (such as whether the camera is moving, as one frame combines information from multiple frames. If static objects in the background appear noticeably blurry, there is a good chance that the camera is moving!):\n'
    }

    # Add descriptions and frame markers for each video type
    for video_type, description in video_descriptions.items():
        config = video_configs.get(video_type)
        if config and config.is_valid():
            question += description
            for i in range(config.num_frames):
                # Calculate frame number
                frame_num = i / config.rate + config.start_frame
                if frame_num == int(frame_num):
                    question = question + f'Frame{int(frame_num)}: {IMAGE_TOKEN}\n'
                else:
                    question = question + f'Frame{frame_num:.2f}: {IMAGE_TOKEN}\n'

    # Add user question
    question += q

    # Build content list
    content = [{'type': 'text', 'text': question}]

    # Add images for each video type
    for video_type in video_descriptions.keys():
        config = video_configs.get(video_type)
        if config and config.is_valid():
            for i in range(config.num_frames):
                image_url_dict = {
                    'max_dynamic_patch': 1,
                    'url': os.path.abspath(f'{config.path}/{i:04d}.jpg')
                }

                if resize_for_memory:
                    image_url_dict['max_pixels'] = 360 * 420

                content.append({
                    'type': 'image_url',
                    'image_url': image_url_dict
                })

    # Create or update conversation history
    if history is None:
        history = [dict(role='user', content=content)]
        response = pipe(history, gen_config=GenerationConfig(max_new_tokens=128, do_sample=False))
        output_text = response.text
        history.append(dict(role='assistant', content=output_text))
    else:
        history.append(dict(role='user', content=content))
        response = pipe(history, gen_config=GenerationConfig(max_new_tokens=128, do_sample=False))
        output_text = response.text
        history.append(dict(role='assistant', content=output_text))

    print(history)
    return output_text, history

def crop_frame(args):
    frame_idx, box, frame_path, output_path = args
    try:
        img = Image.open(frame_path)
        # Add padding to the box
        padding = 20  # pixels
        x1 = max(0, box[0] - padding)
        y1 = max(0, box[1] - padding)
        x2 = min(img.width, box[2] + padding)
        y2 = min(img.height, box[3] + padding)
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_path = f"{output_path}/{frame_idx:04d}.jpg"
        cropped_img.save(cropped_path)
        return frame_idx, True
    except Exception as e:
        import traceback
        print(f"Error cropping frame {frame_idx}: {traceback.format_exc()}")
        return frame_idx, False

def visual_spotlight(args):
    frame_idx, box, frame_path, output_path = args
    try:
        img = Image.open(frame_path)
        x1 = max(0, box[0])
        y1 = max(0, box[1])
        x2 = min(img.width, box[2])
        y2 = min(img.height, box[3])

        # Create a darkened copy of the original image
        darkened_img = img.copy()
        darken_factor = 0.9
        darkened_img = Image.blend(darkened_img, Image.new('RGB', darkened_img.size, (0, 0, 0)), darken_factor)

        # Create a mask for the highlighted region
        mask = Image.new('L', img.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([x1, y1, x2, y2], fill=255)

        # Apply the original image to the highlighted region
        result = darkened_img.copy()
        result.paste(img, (0, 0), mask)

        # Save the result
        result_path = f"{output_path}/{frame_idx:04d}.jpg"
        result.save(result_path)

        return frame_idx, True
    except Exception as e:
        import traceback
        print(f"Error processing frame {frame_idx}: {traceback.format_exc()}")
        return frame_idx, False

def save_frames(video_path, return_msg=False, fps_sample_rate=1, dense=False, num_segs=num_segs):
    """Save video frames to disk using the same sampling logic as load_video

    Args:
        video_path (str): Path to the video file
        return_msg (bool): Whether to return a message with time information

    Returns:
        tuple: Directory path where frames are saved, frame width, frame height, number of frames, and optionally a message with time information
    """

    clip = VideoFileClip(video_path)
    max_frame = int(clip.fps * clip.duration)
    fps = clip.fps

    video_name = video_path.split("/")[-1].split(".")[0]
    save_dir = f'./{restore_name}/{video_name}_dense' if dense else f'./{restore_name}/{video_name}'
    os.makedirs(save_dir, exist_ok=True)

    if do_fps:
        start, end = 0, clip.duration
        start_idx = max(0, round(start * fps))
        end_idx = min(round(end * fps), max_frame)

        sample_interval = int(fps / fps_sample_rate)
        frame_indices = list(range(start_idx, end_idx + 1, sample_interval))
    else:
        frame_indices = get_index(None, fps, max_frame, first_idx=0, num_segments=num_segs)

    frame_width, frame_height = clip.size

    for i, frame_idx in enumerate(frame_indices):
        time = frame_idx / fps
        if time <= clip.duration:
            img = Image.fromarray(np.array(clip.get_frame(time)).astype('uint8')).convert('RGB')
            img.save(f"{save_dir}/{i:04d}.jpg")

    clip.close()

    if return_msg:
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds. "
        return os.path.abspath(save_dir), frame_width, frame_height, len(frame_indices), msg, frame_indices
    else:
        return os.path.abspath(save_dir), frame_width, frame_height, len(frame_indices), frame_indices


def run(pipe, video_path, q, has_cm=True, has_om=True, rc=False):
    from Grounded_SAM_2.new_track_utils import run_track
    detection_boxes_raw = None

    assert (has_cm and not has_om) or (not has_cm and has_om)
    # --------------Step 1--------------

    frames_path, frame_width, frame_height, len_frames, time_info, frame_indices = save_frames(video_path, return_msg=True)

    # First detection for action objects
    if has_om:
        question = f'''I have a question: "{q}". I need you to analyze the above question step by step. **In this step, you don't need to directly answer the question.**

Please provide your response in the following JSON format **without any comment**:
{{
    "action_objects": ["object1", "object2", ...],
}}

For the "action_objects" field, provide a list of strings, each describing a specific entity that is involved in the main action or motion. Each entity should be a single object or a group of objects. For example, if the question is about a person eating, include both the person and the rice bowl. If the question is about object motion, make sure to include both the moving objects (actors/performers) and the objects they interact with or affect. You can also provide fine-grained components of larger objects when relevant (e.g., not just "person" but also "person's head", "person's hands", etc.). Each string represents a different object. All items must be physical entities that can be visually identified, not abstract concepts. **Only keep the moving objects that are highly relevant to the question and reduce the background objects.** You must provide at least one action_object.

Watch the video, then provide the JSON response as described above.'''
        while True:
            try:
                out, history = ask_mllm(pipe, question, frames_path, history=None, len_frames=len_frames)
                # Try to parse the JSON response
                response_json = parser_json(out)
                action_objects = response_json["action_objects"]
                # assert len(action_objects) > 0
                break  # If parsing succeeds, exit the loop
            except Exception as e:
                print("\033[93m{str(e)}\033[0m")


        # --------------Step 2 part1--------------

        # First detection for action objects
        if action_objects:
            print("I am waiting for tracking agent's result for action objects...")
            detection_boxes_raw = run_track(frames_path, action_objects)

    # --------------Step 2 part2--------------
    spotlight_func = crop_frame if rc else visual_spotlight

    if (not do_fps and num_segs_dense != num_segs) or (do_fps and new_rate != 1):
        dense_frames_path, _, _, dense_len_frames, _ = save_frames(video_path, return_msg=True, fps_sample_rate=new_rate, dense=True, num_segs=num_segs_dense)
    else:
        dense_frames_path = frames_path
        dense_len_frames = len_frames
    # Calculate union bounding box for action objects
    action_dir = None
    if detection_boxes_raw is not None:
        action_dir = f"{frames_path}_action"
        os.makedirs(action_dir, exist_ok=True)

        # Cluster detection boxes in temporal dimension
        clusters = cluster_temporal_boxes(detection_boxes_raw, len_frames, frame_width, frame_height)

        # Process each frame using the union box of its corresponding time segment
        for frame_idx in range(dense_len_frames):
            frame_path = f"{dense_frames_path}/{frame_idx:04d}.jpg"
            if not os.path.exists(frame_path):
                continue

            # Find the cluster that the current frame belongs to
            current_cluster = None
            for cluster in clusters:
                if cluster['start_frame'] <= frame_idx/new_rate <= cluster['end_frame']:
                    current_cluster = cluster
                    break

            if current_cluster is not None:
                union_box = current_cluster['union_box']
                # Add padding
                padding = 20  # pixels
                min_x1 = max(0, union_box[0] - padding)
                min_y1 = max(0, union_box[1] - padding)
                max_x2 = min(frame_width, union_box[2] + padding)
                max_y2 = min(frame_height, union_box[3] + padding)

                spotlight_func((frame_idx, [min_x1, min_y1, max_x2, max_y2], frame_path, action_dir))
            else:
                assert False
    else:
        action_dir = frames_path

    # motion blur
    blur_frames_path = None
    if has_cm:
        blur_frames_path = f"./{restore_name}/{os.path.basename(video_path).split('.')[0]}_blur"
        print(f"Applying motion trails to video: {video_path}")
        apply_motion_trails(video_path, blur_frames_path, trail_indices=frame_indices)
        print(f"Saved blur frames at: {blur_frames_path}\n")

    # -----final-----
    final_prompt = f"""Here is the question: \"{q}\"."""
    final_prompt += """\n\nReply based on the above information. Answer only the answer letter without showing your process."""

    # prompt template
    video_configs = {}

    if has_cm:
        video_configs['motion_blur'] = VideoConfig(blur_frames_path, len_frames, start_frame=0, rate=1)
        final_out, _ = ask_mllm_multi(pipe, final_prompt, video_configs=video_configs)
    elif has_om:
        video_configs['spotlight'] = VideoConfig(action_dir, dense_len_frames, start_frame=0, rate=new_rate)
        final_out, _ = ask_mllm_multi(pipe, final_prompt, video_configs=video_configs)
    else:
        print("\033[93mSomething error...\033[0m")
        assert False

    return final_out

def read_video_info(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
    return data

def cluster_temporal_boxes(detection_boxes, len_frames, frame_width, frame_height):
    """
    Temporal clustering for detection boxes
    Args:
        detection_boxes: Dict[int, Dict[int, List[float]]] - First key is frame index, second key is object ID
        len_frames: int - Total number of frames
        frame_width: int - Frame width
        frame_height: int - Frame height
    """
    print("\n=== Starting Temporal Clustering ===")

    # Motion related thresholds
    max_center_shift = min(frame_width, frame_height) * 0.3
    min_cluster_frames = 3
    print(f"Parameters: max_center_shift={max_center_shift:.2f}, min_cluster_frames={min_cluster_frames}")

    # Collect valid boxes for each frame
    frame_to_boxes = {i: [] for i in range(len_frames)}

    print("\nCollecting valid boxes for each frame...")
    for frame_idx in detection_boxes:
        for obj_id, box in detection_boxes[frame_idx].items():
            # Skip invalid boxes (all zeros)
            if box[0] == 0 and box[1] == 0 and box[2] == 0 and box[3] == 0:
                continue
            frame_to_boxes[frame_idx].append(box)

    # Temporal clustering based on center displacement
    clusters = []
    current_cluster = {
        'start_frame': 0,
        'end_frame': 0,
        'boxes': [],
        'center': None,
        'all_centers': []
    }

    print("\nStarting clustering process...")
    last_valid_center = None
    last_valid_box = None

    for frame_idx in range(len_frames):
        current_boxes = frame_to_boxes[frame_idx] if frame_idx in frame_to_boxes else []

        if current_boxes:
            current_center = np.mean([[((b[0]+b[2])/2, (b[1]+b[3])/2) for b in current_boxes]], axis=0)[0]
            last_valid_center = current_center
            last_valid_box = current_boxes[0]  # Save the valid box
        elif last_valid_center is not None:
            current_center = last_valid_center
            if last_valid_box is not None:  # Use the last valid box
                current_boxes = [last_valid_box]
        else:
            # Include this frame even if there's no valid box
            # Use the entire frame as a box
            current_boxes = [[0, 0, frame_width, frame_height]]
            current_center = np.array([frame_width/2, frame_height/2])
            last_valid_center = current_center
            last_valid_box = current_boxes[0]

        if current_cluster['center'] is None:
            current_cluster['center'] = current_center
            current_cluster['boxes'].extend(current_boxes)
            current_cluster['all_centers'].append(current_center)
            current_cluster['end_frame'] = frame_idx
            print(f"\nStarting new cluster at frame {frame_idx}")
            continue

        prev_center = current_cluster['center']
        displacement = np.sqrt(np.sum((current_center - prev_center)**2))

        if displacement > max_center_shift and len(current_cluster['boxes']) >= min_cluster_frames:
            cluster_boxes = current_cluster['boxes']
            union_box = [
                min(box[0] for box in cluster_boxes),
                min(box[1] for box in cluster_boxes),
                max(box[2] for box in cluster_boxes),
                max(box[3] for box in cluster_boxes)
            ]

            # Ensure the box has a reasonable size
            if union_box[2] - union_box[0] < 10 or union_box[3] - union_box[1] < 10:
                # If the box is too small, enlarge it
                center_x = (union_box[0] + union_box[2]) / 2
                center_y = (union_box[1] + union_box[3]) / 2
                width = max(union_box[2] - union_box[0], 100)
                height = max(union_box[3] - union_box[1], 100)
                union_box = [
                    max(0, center_x - width/2),
                    max(0, center_y - height/2),
                    min(frame_width, center_x + width/2),
                    min(frame_height, center_y + height/2)
                ]

            clusters.append({
                'start_frame': current_cluster['start_frame'],
                'end_frame': current_cluster['end_frame'],
                'union_box': union_box
            })
            print(f"\nCluster completed:")
            print(f"  Frames: {current_cluster['start_frame']} - {current_cluster['end_frame']}")
            print(f"  Union box: [x1={union_box[0]:.1f}, y1={union_box[1]:.1f}, x2={union_box[2]:.1f}, y2={union_box[3]:.1f}]")
            print(f"  Box size: {union_box[2]-union_box[0]:.1f} x {union_box[3]-union_box[1]:.1f}")

            current_cluster = {
                'start_frame': frame_idx,
                'end_frame': frame_idx,
                'boxes': current_boxes,
                'center': current_center,
                'all_centers': [current_center]
            }
            print(f"\nStarting new cluster at frame {frame_idx} (displacement={displacement:.1f} > threshold={max_center_shift:.1f})")
        else:
            current_cluster['boxes'].extend(current_boxes)
            current_cluster['end_frame'] = frame_idx
            current_cluster['all_centers'].append(current_center)
            alpha = 0.7
            current_cluster['center'] = alpha * prev_center + (1-alpha) * current_center

    # Handle the last cluster
    if current_cluster['boxes']:
        cluster_boxes = current_cluster['boxes']
        union_box = [
            min(box[0] for box in cluster_boxes),
            min(box[1] for box in cluster_boxes),
            max(box[2] for box in cluster_boxes),
            max(box[3] for box in cluster_boxes)
        ]

        # Ensure the box has a reasonable size
        if union_box[2] - union_box[0] < 10 or union_box[3] - union_box[1] < 10:
            # If the box is too small, enlarge it
            center_x = (union_box[0] + union_box[2]) / 2
            center_y = (union_box[1] + union_box[3]) / 2
            width = max(union_box[2] - union_box[0], 100)
            height = max(union_box[3] - union_box[1], 100)
            union_box = [
                max(0, center_x - width/2),
                max(0, center_y - height/2),
                min(frame_width, center_x + width/2),
                min(frame_height, center_y + height/2)
            ]

        clusters.append({
            'start_frame': current_cluster['start_frame'],
            'end_frame': current_cluster['end_frame'] + 1,
            'union_box': union_box
        })
        print(f"\nFinal cluster completed:")
        print(f"  Frames: {current_cluster['start_frame']} - {current_cluster['end_frame'] + 1}")
        print(f"  Union box: [x1={union_box[0]:.1f}, y1={union_box[1]:.1f}, x2={union_box[2]:.1f}, y2={union_box[3]:.1f}]")
        print(f"  Box size: {union_box[2]-union_box[0]:.1f} x {union_box[3]-union_box[1]:.1f}")

    # Check for gaps between clusters and fill them using the union of adjacent clusters' boxes
    sorted_clusters = sorted(clusters, key=lambda x: x['start_frame'])
    filled_clusters = []

    for i in range(len(sorted_clusters)):
        filled_clusters.append(sorted_clusters[i])

        # Check for gap with the next cluster
        if i < len(sorted_clusters) - 1:
            current_end = sorted_clusters[i]['end_frame']
            next_start = sorted_clusters[i+1]['start_frame']

            if current_end < next_start - 1:
                # There's a gap, create a transition cluster
                current_box = sorted_clusters[i]['union_box']
                next_box = sorted_clusters[i+1]['union_box']

                # Calculate the union of the two boxes
                union_box = [
                    min(current_box[0], next_box[0]),
                    min(current_box[1], next_box[1]),
                    max(current_box[2], next_box[2]),
                    max(current_box[3], next_box[3])
                ]

                transition_cluster = {
                    'start_frame': current_end + 1,
                    'end_frame': next_start - 1,
                    'union_box': union_box
                }

                filled_clusters.append(transition_cluster)
                print(f"\nAdded transition cluster:")
                print(f"  Frames: {current_end + 1} - {next_start - 1}")
                print(f"  Union box: [x1={union_box[0]:.1f}, y1={union_box[1]:.1f}, x2={union_box[2]:.1f}, y2={union_box[3]:.1f}]")

    # Check if the first frame is 0, if not, add a cluster from 0 to the start of the first cluster
    if filled_clusters and filled_clusters[0]['start_frame'] > 0:
        first_box = filled_clusters[0]['union_box']
        initial_cluster = {
            'start_frame': 0,
            'end_frame': filled_clusters[0]['start_frame'] - 1,
            'union_box': first_box
        }
        filled_clusters.insert(0, initial_cluster)
        print(f"\nAdded initial cluster:")
        print(f"  Frames: 0 - {filled_clusters[1]['start_frame'] - 1}")
        print(f"  Union box: [x1={first_box[0]:.1f}, y1={first_box[1]:.1f}, x2={first_box[2]:.1f}, y2={first_box[3]:.1f}]")

    # Check if the last frame is len_frames-1, if not, add a cluster from the end of the last cluster to len_frames-1
    if filled_clusters and filled_clusters[-1]['end_frame'] < len_frames - 1:
        last_box = filled_clusters[-1]['union_box']
        final_cluster = {
            'start_frame': filled_clusters[-1]['end_frame'] + 1,
            'end_frame': len_frames - 1,
            'union_box': last_box
        }
        filled_clusters.append(final_cluster)
        print(f"\nAdded final cluster:")
        print(f"  Frames: {filled_clusters[-2]['end_frame'] + 1} - {len_frames - 1}")
        print(f"  Union box: [x1={last_box[0]:.1f}, y1={last_box[1]:.1f}, x2={last_box[2]:.1f}, y2={last_box[3]:.1f}]")

    # Use the filled clusters
    clusters = filled_clusters

    # Verify coverage
    covered_frames = set()
    for cluster in clusters:
        covered_frames.update(range(cluster['start_frame'], cluster['end_frame'] + 1))

    print(f"\nClustering completed. Total clusters: {len(clusters)}")
    print(f"Frame coverage: {len(covered_frames)}/{len_frames} frames")
    if len(covered_frames) < len_frames:
        print("Warning: Some frames are not covered!")
        uncovered = set(range(len_frames)) - covered_frames
        print(f"Uncovered frames: {sorted(list(uncovered))}")
        assert False
    else:
        print("Success: All frames are covered!")

    print("\nCluster summary:")
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i+1}:")
        print(f"  Frames: {cluster['start_frame']} - {cluster['end_frame']} ({cluster['end_frame']-cluster['start_frame']+1} frames)")
        box = cluster['union_box']
        print(f"  Union box: [x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}]")
        print(f"  Box size: {box[2]-box[0]:.1f} x {box[3]-box[1]:.1f}")

    return clusters

def process_chunk_distributed(local_rank, data_chunk, temp_output_path):
    print(f"Process {local_rank} starting work on CUDA:{local_rank}, processing {len(data_chunk)} videos")

    pipe = pipeline(
        checkpoint_path,
        backend_config=PytorchEngineConfig(
            tp=1,
            cache_max_entry_count=0.4
        ),
        chat_template_config=ChatTemplateConfig('internvl2_5'),
    )

    # Prepare temporary output file
    process_output_file = f"{temp_output_path}_{local_rank}.jsonl"

    for i, item in enumerate(tqdm(data_chunk, desc=f"GPU {local_rank} processing videos")):
        value_list = []
        print(f"GPU {local_rank} processing: {item}")
        video_path = item["video_path"]
        for q in item['qa']:
            answer = q['answer']
            if answer == "NA":
                # judge = None
                continue
            start = q["start"]
            end = q["end"]
            video_path1 = os.path.join(args.motionbench_pos, "MotionBench", "public-dataset", video_path)
            video_path2 = os.path.join(args.motionbench_pos, "MotionBench", "self-collected", video_path)
            video_path = video_path1 if os.path.exists(video_path1) else video_path2

            question = f"{q['question']}\nAnswer with only the option letter."
            has_cm = item["question_type"] == "Camera Motion"
            has_om = not has_cm
            rc = item["question_type"] == "Repetition Count"
            mllm_response = run(pipe, video_path, question, has_cm, has_om, rc)
            if answer in mllm_response:
                judge = True
            else:
                judge = False

            print(question)
            out = {'question':q['question'], 'question_type':item['question_type'], 'correct_answer':answer, 'output':mllm_response, 'judge':judge}
            if judge is not None:
                if judge:
                    print(f"\033[1;32m{out}\033[0m")
                else:
                    print(f"\033[1;31m{out}\033[0m")
            else:
                print(out)
            value_list.append(out)

        basename, _ = os.path.splitext(os.path.basename(video_path))

        new_item = {basename:value_list}

        with open(process_output_file, 'a', encoding='utf-8') as output_file:
            json.dump(new_item, output_file, ensure_ascii=False)
            output_file.write('\n')

def merge_results(temp_output_path, world_size, final_output_path):
    """Merge all temporary result files into the final output file"""
    with open(final_output_path, 'w', encoding='utf-8') as final_file:
        for local_rank in range(world_size):
            temp_file = f"{temp_output_path}_{local_rank}.jsonl"
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        final_file.write(line)
                # Optionally delete temporary files
                os.remove(temp_file)
    print(f"All results have been merged into {final_output_path}")


def main():
    file_path = os.path.join(args.motionbench_pos, "MotionBench/video_info.meta.jsonl")
    data = read_video_info(file_path)

    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    world_size = dist.get_world_size()
    chunk_size = len(data) // world_size
    start_idx = local_rank * chunk_size
    end_idx = start_idx + chunk_size if local_rank < world_size - 1 else len(data)
    data_chunk = data[start_idx:end_idx]

    # Create temporary output path
    temp_output_path = os.path.splitext(output_filename)[0] + "_temp"

    process_chunk_distributed(local_rank, data_chunk, temp_output_path)

    dist.barrier()

    if local_rank == 0:
        # Merge results
        merge_results(temp_output_path, world_size, output_filename)

    dist.destroy_process_group()



if __name__ == "__main__":
    main()