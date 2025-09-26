import numpy as np
import json
from PIL import Image, ImageDraw
import socket
import pickle
import ast
import os
import time
import torch
import random
import argparse
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.utils import *
from typing import List, Dict, Any
from moviepy import VideoFileClip
import torch.multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser(description='Parallel evaluation of the FAVOR video dataset')
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(),
                        help='Number of GPUs to use')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--stage', type=int)
    parser.add_argument('--favor_pos', type=str,
                        help='Path to data file')
    parser.add_argument('--output_filename', type=str, default="motionchat_favor.jsonl",
                        help='Output filename')
    parser.add_argument('--resume', type=int, default=0,
                        help='Index to resume evaluation from')
    parser.add_argument('--restore_name', type=str, default='restore_qwenfavor',
                        help='Directory name to save video frames')
    parser.add_argument('--num_segs', type=int, default=16,
                        help='Number of video segments')
    parser.add_argument('--do_fps', type=bool, default=True,
                        help='Whether to use FPS sampling')
    return parser.parse_args()

def load_model(checkpoint_path, gpu_id):
    device = f"cuda:{gpu_id}"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    return model, processor

def save_frames(video_path, restore_name, do_fps=True, num_segs=16, return_msg=False, fps_sample_rate=1, dense=False):
    """Save video frames to disk using the same sampling logic as load_video"""

    clip = VideoFileClip(video_path)
    max_frame = int(clip.fps * clip.duration)
    fps = clip.fps

    video_name = video_path.split("/")[-1].split(".")[0]
    save_dir = f'./temp/{restore_name}/{video_name}_dense' if dense else f'./temp/{restore_name}/{video_name}'
    os.makedirs(save_dir, exist_ok=True)

    if do_fps:
        start, end = 0, clip.duration
        start_idx = max(0, round(start * fps))
        end_idx = min(round(end * fps), max_frame - 1)  # Ensure not to exceed max frame count
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
        return os.path.abspath(save_dir), frame_width, frame_height, len(frame_indices), msg
    else:
        return os.path.abspath(save_dir), frame_width, frame_height, len(frame_indices)

def ask_mllm(model, processor, q, frames_path, len_frames, gpu_id):
    content = []

    for i in range(len_frames):
        content.append({
            "type": "image",
            "image": f"{frames_path}/{i:04d}.jpg",
            "max_pixels": 360 * 420,
        })

    content.append({"type": "text", "text": q})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    device = f"cuda:{gpu_id}"
    inputs = inputs.to(device)

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128, temperature=0, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

def run(model, processor, video_path, q, restore_name, do_fps, num_segs, gpu_id):
    final_prompt = q

    frames_path, frame_width, frame_height, len_frames, time_info = save_frames(
        video_path, restore_name, do_fps, num_segs, return_msg=True
    )
    final_out = ask_mllm(model, processor, final_prompt, frames_path, len_frames, gpu_id)

    return final_out

def process_chunk(gpu_id, data_chunk, args, output_lock, temp_output_path):
    # Set CUDA device
    torch.cuda.set_device(gpu_id)
    print(f"Process {gpu_id} started working on CUDA:{gpu_id}, processing {len(data_chunk)} videos")

    # Load model
    model, processor = load_model(args.checkpoint_path, gpu_id)

    # Prepare temporary output file
    process_output_file = f"{temp_output_path}_{gpu_id}.jsonl"

    # Get list of already processed videos
    output_dict = {}
    if os.path.exists(args.output_filename):
        with open(args.output_filename, 'r') as f:
            for line in f:
                try:
                    output_dict.update(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing output file: {e}")

    for item in tqdm(data_chunk, desc=f"GPU {gpu_id} processing videos"):
        video_name = item["video_name"]

        # Skip already processed videos
        if video_name.split('.')[0] in output_dict:
            print(f"GPU {gpu_id} - Skipping already processed video: {video_name}")
            continue

        video_path = os.path.join(args.favor_pos, "videos/FAVOR-Bench", video_name)
        if not os.path.exists(video_path):
            print(f"GPU {gpu_id} - Video does not exist: {video_path}")
            continue

        value_list = []

        for question in tqdm(item['questions'], desc=f"GPU {gpu_id} - Processing questions for {video_name}", leave=False):
            task_type = question['task_type']
            correct_answer = question['correct_answer']
            options = question['options']
            option_labels = [chr(i) for i in range(65, 65 + len(options))]  # Generate option labels A, B, C, etc.
            option_dict = {label: option for label, option in zip(option_labels, options)}
            formatted_options = [f"{label}. {option}" for label, option in option_dict.items()]
            formatted_options_str = '\n'.join(formatted_options)

            # prompt = f"Carefully watch the video and pay attention to temporal dynamics in this video, focusing on the camera motions, actions, activities, and interactions. Based on your observations, select the best option that accurately addresses the question.\n{question['question']}\nYou can only respond with the answer among {formatted_options_str}"
            if args.stage == 2:
                prompt = f"Describe the movements of the main subjects in this clip.\n{question['question']}\n{formatted_options_str}\nPlease respond with only the letter of the correct answer."
            else:
                prompt = f"{question['question']}\n{formatted_options_str}"

            try:
                output_text = run(model, processor, video_path, prompt,
                                 args.restore_name, args.do_fps, args.num_segs, gpu_id)

                correct_answer_label = next((label for label, option in option_dict.items() if option == correct_answer), None)
                if correct_answer_label and correct_answer_label[0].lower() == output_text[0].lower():
                    judge = True
                else:
                    judge = False

                out = {'task_type': task_type, 'prompt': prompt, 'correct_answer': correct_answer_label, 'output': output_text, 'judge': judge}
                value_list.append(out)

                if judge:
                    print(f"GPU {gpu_id} - \033[1;32m{out}\033[0m")
                else:
                    print(f"GPU {gpu_id} - \033[1;31m{out}\033[0m")
            except Exception as e:
                print(f"GPU {gpu_id} - Error processing question: {e}")
                out = {'task_type': task_type, 'correct_answer': correct_answer,
                      'output': f"ERROR: {str(e)}", 'judge': False}
                value_list.append(out)

        basename, _ = os.path.splitext(os.path.basename(video_path))
        new_item = {basename:value_list}

        # Write to temporary file
        with open(process_output_file, 'a', encoding='utf-8') as output_file:
            json.dump(new_item, output_file, ensure_ascii=False)
            output_file.write('\n')

    print(f"GPU {gpu_id} processing complete")

def merge_results(temp_output_path, num_gpus, final_output_path):
    """Merge all temporary result files into the final output file"""
    with open(final_output_path, 'a', encoding='utf-8') as final_file:
        for gpu_id in range(num_gpus):
            temp_file = f"{temp_output_path}_{gpu_id}.jsonl"
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        final_file.write(line)
                # Optional: delete temporary file
                os.remove(temp_file)
    print(f"All results have been merged into {final_output_path}")

def main():
    args = parse_args()

    # Ensure output directory exists
    os.makedirs("temp/", exist_ok=True)
    os.makedirs(f"./temp/{args.restore_name}", exist_ok=True)

    # Read data
    with open(os.path.join(args.favor_pos, "video_perspective.json"), 'r') as f:
        data = json.load(f)

    if args.resume > 0:
        data = data[args.resume:]

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f"Using {num_gpus} GPUs for parallel processing")

    # Split data
    chunk_size = len(data) // num_gpus
    data_chunks = []

    for i in range(num_gpus):
        if i == num_gpus - 1:  # The last chunk may be larger
            data_chunks.append(data[i * chunk_size:])
        else:
            data_chunks.append(data[i * chunk_size:(i + 1) * chunk_size])

    # Create temporary output path
    temp_output_path = os.path.splitext(args.output_filename)[0] + "_temp"

    # Multi-process handling
    mp.set_start_method('spawn', force=True)
    output_lock = mp.Lock()
    processes = []

    try:
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=process_chunk,
                args=(gpu_id, data_chunks[gpu_id], args, output_lock, temp_output_path)
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Merge results
        merge_results(temp_output_path, num_gpus, args.output_filename)

    except Exception as e:
        print(f"Error in parallel processing: {e}")
        # Terminate all processes
        for p in processes:
            if p.is_alive():
                p.terminate()
        raise e

if __name__ == "__main__":
    main()