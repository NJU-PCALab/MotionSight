#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
from tqdm import tqdm
import t2v_metrics
import numpy as np
from PIL import Image
import argparse
import torch
from accelerate import Accelerator
from moviepy.editor import VideoFileClip
import glob

def get_score(model, images, text, temp_dir=None):
    """Calculate VQA score for a set of frames and text"""
    # Create process-specific temp directory
    if temp_dir is None:
        temp_dir = f'temp_{os.getpid()}'
    os.makedirs(temp_dir, exist_ok=True)
    
    image_paths = []
    try:
        for i, img in enumerate(images):
            path = os.path.join(temp_dir, f'image_{i}.png')
            Image.fromarray(img).save(path, 'PNG')
            image_paths.append(path)
        score = model(images=image_paths, texts=[text])
        return score.cpu().numpy().mean()
    except Exception as e:
        print(f"Error in get_score: {e}")
        return float('nan')
    finally:
        # Always clean up temp files
        for path in image_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {path}: {e}")
        # Try to remove the temp dir if it's empty
        try:
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass

def calculate_vqa_score(model, video_path, motion_type, mode="ours", num_samples=6):
    """Calculate VQA score for a video and text using moviepy"""
    # Determine text prompt based on mode and motion type
    if mode == "original":
        if motion_type == "object":
            text = "original_object_motion"
        else:  # "camera"
            text = "original_camera_motion"
    else:  # "ours"
        if motion_type == "object":
            text = "object motion"
        else:  # "camera"
            text = "camera motion"
            
    try:
        # Load video with moviepy
        with VideoFileClip(video_path) as clip:
            # Get video duration and calculate sampling times
            duration = clip.duration
            if duration <= 0:
                print(f"Video has invalid duration: {video_path}")
                return float('nan')
                
            # Sample frames at regular intervals
            frames = []
            for i in range(num_samples):
                t = min(i * duration / num_samples, duration - 0.1)  # Ensure we don't sample past the end
                frame = clip.get_frame(t)  # This returns RGB format by default
                frames.append(frame)
                
        if not frames:
            return float('nan')
        return get_score(model, frames, text)
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return float('nan')

def get_all_dataset_items(results_dir, datasets_to_process=None):
    """Get all items to process from specified datasets"""
    all_items = []
    
    # Get list of directories
    available_datasets = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    # Filter datasets if specified
    if datasets_to_process:
        dataset_names = [d for d in available_datasets if d in datasets_to_process]
    else:
        dataset_names = available_datasets
    
    for dataset_name in dataset_names:
        dataset_dir = os.path.join(results_dir, dataset_name)
        jsonl_path = os.path.join(dataset_dir, f"{dataset_name}_processed.jsonl")
        
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} not found")
            continue
            
        # Read all items from JSONL
        dataset_items = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    video_file = data.get('video_file')
                    if video_file:
                        dataset_items.append({
                            'dataset_name': dataset_name,
                            'dataset_dir': dataset_dir,
                            'video_file': video_file
                        })
                except Exception as e:
                    print(f"Error parsing line in {jsonl_path}: {e}")
        
        all_items.extend(dataset_items)
        print(f"Added {len(dataset_items)} items from {dataset_name}")
        
    return all_items

def save_results(results, dataset_dir, dataset_name, process_id, batch_id=0):
    """Save results to CSV file"""
    # Use process_id and batch_id in filename to avoid conflicts
    temp_csv_path = os.path.join(dataset_dir, f"{dataset_name}_motion_scores_proc{process_id}_batch{batch_id}.csv")
    final_csv_path = os.path.join(dataset_dir, f"{dataset_name}_motion_scores.csv")
    
    # Write to process-specific file
    with open(temp_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_path', 'object_motion_score', 'camera_motion_score'])
        for result in results:
            writer.writerow([
                result['video_path'],
                result['object_motion_score'],
                result['camera_motion_score']
            ])
    
    print(f"Process {process_id} saved batch {batch_id} results to {temp_csv_path}, length {len(results)}")
    return temp_csv_path, final_csv_path

def merge_results(all_temp_files, final_csv_path):
    """Merge all temporary CSV files into a final CSV file"""
    print(f"Starting to merge {len(all_temp_files)} files into {final_csv_path}")
    
    # 计算所有临时文件中的总行数（不包括标题行）
    total_rows = 0
    for temp_file in all_temp_files:
        if os.path.exists(temp_file):
            with open(temp_file, 'r', encoding='utf-8', newline='') as infile:
                # 减去一行标题
                total_rows += sum(1 for _ in infile) - 1
    
    print(f"Found {total_rows} total entries across all temporary files")
    
    # 打开最终CSV文件并写入所有数据
    with open(final_csv_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['video_path', 'object_motion_score', 'camera_motion_score'])
        
        merged_count = 0
        for temp_file in all_temp_files:
            if os.path.exists(temp_file):
                # 读取每个临时文件并合并
                with open(temp_file, 'r', encoding='utf-8', newline='') as infile:
                    reader = csv.reader(infile)
                    next(reader)  # Skip header
                    for row in reader:
                        writer.writerow(row)
                        merged_count += 1
                
                # 删除临时文件
                os.remove(temp_file)
                print(f"Removed temporary file: {temp_file}")
    
    print(f"Successfully merged {merged_count} entries into {final_csv_path}")
    return merged_count

# 添加清理临时文件的函数
def cleanup_temp_files(dataset_dir, dataset_name):
    """清理指定数据集目录中的所有临时文件"""
    pattern = f"{dataset_name}_motion_scores_proc*_batch*.csv"
    temp_files = glob.glob(os.path.join(dataset_dir, pattern))
    
    if temp_files:
        print(f"Cleaning up {len(temp_files)} remaining temporary files for {dataset_name}...")
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                print(f"Removed: {temp_file}")
            except Exception as e:
                print(f"Failed to remove {temp_file}: {e}")
    return len(temp_files)

def cleanup_temp_directories():
    """清理所有临时图像目录"""
    # 查找所有以temp_开头的目录
    temp_dirs = glob.glob('temp_*')
    if temp_dirs:
        print(f"Cleaning up {len(temp_dirs)} temporary image directories...")
        for temp_dir in temp_dirs:
            if os.path.isdir(temp_dir):
                try:
                    # 首先删除目录中的所有文件
                    for file in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    # 然后删除目录
                    os.rmdir(temp_dir)
                    print(f"Removed temporary directory: {temp_dir}")
                except Exception as e:
                    print(f"Failed to remove temporary directory {temp_dir}: {e}")
    return len(temp_dirs)

def main():
    parser = argparse.ArgumentParser(description="Calculate VQA scores for object and camera motion")
    parser.add_argument('--results_dir', type=str, required=True, help="Path to results_cleaned directory")
    parser.add_argument('--video_root', type=str, required=True, help="Path to video root directory")
    parser.add_argument('--model', type=str, default='clip-flant5-xxl', help="VQA model name")
    parser.add_argument('--save_interval', type=int, default=20, 
                        help="Save results after processing this many videos. Smaller values are safer but create more temp files.")
    parser.add_argument('--datasets', type=str, nargs='+', default=None, 
                         help="Specific dataset folders to process (e.g., 'SSV2 Charades'). If not specified, all folders will be processed.")
    parser.add_argument('--mode', type=str, choices=['original', 'ours'], default='ours',
                       help="Evaluation mode: 'original' uses original_object_motion/original_camera_motion, 'ours' uses object motion/camera motion")
    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator()
    is_main_process = accelerator.is_main_process
    process_index = accelerator.process_index
    
    if is_main_process:
        print(f"Running with {accelerator.num_processes} processes")
        print(f"Evaluation mode: {args.mode}")
        if args.datasets:
            print(f"Will process these datasets: {', '.join(args.datasets)}")
        else:
            print("Will process all available datasets")
            
    # Get all items to process
    if is_main_process:
        all_items = get_all_dataset_items(args.results_dir, args.datasets)
        print(f"Total items to process: {len(all_items)}")
    else:
        all_items = get_all_dataset_items(args.results_dir, args.datasets)
    
    # Distribute items across processes
    items_per_process = len(all_items) // accelerator.num_processes
    remainder = len(all_items) % accelerator.num_processes
    
    start_idx = process_index * items_per_process + min(process_index, remainder)
    if process_index < remainder:
        items_per_process += 1
    end_idx = start_idx + items_per_process
    
    process_items = all_items[start_idx:end_idx]
    
    print(f"Process {process_index} handling {len(process_items)} items from index {start_idx} to {end_idx-1}")
    
    # Initialize model on the current device
    model = t2v_metrics.VQAScore(model=args.model, device=accelerator.device)
    
    # Process items
    results_by_dataset = {}
    temp_files_by_dataset = {}
    batch_counters = {}  # 添加批次计数器
    
    # Process items
    for i, item in enumerate(tqdm(process_items, desc=f"Process {process_index}")):
        dataset_name = item['dataset_name']
        dataset_dir = item['dataset_dir']
        video_file = item['video_file']
        
        # Initialize results array for this dataset if not exists
        if dataset_name not in results_by_dataset:
            results_by_dataset[dataset_name] = []
            temp_files_by_dataset[dataset_name] = []
            batch_counters[dataset_name] = 0  # 初始化批次计数器
        
        # Construct full video path
        video_path = os.path.join(args.video_root, video_file)
        
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            continue
        
        # Calculate scores for both motion types using the specified mode
        object_score = calculate_vqa_score(model, video_path, "object", args.mode)
        camera_score = calculate_vqa_score(model, video_path, "camera", args.mode)
        
        # Store result
        results_by_dataset[dataset_name].append({
            'video_path': video_path,
            'object_motion_score': object_score,
            'camera_motion_score': camera_score
        })
        
        # Save results periodically
        if (i + 1) % args.save_interval == 0 or i == len(process_items) - 1:
            for ds_name, results in results_by_dataset.items():
                if results:
                    ds_dir = next((item['dataset_dir'] for item in process_items if item['dataset_name'] == ds_name), None)
                    if ds_dir:
                        # 使用批次计数器生成唯一的临时文件名
                        temp_file, final_file = save_results(results, ds_dir, ds_name, process_index, batch_counters[ds_name])
                        temp_files_by_dataset[ds_name].append(temp_file)
                        # 增加批次计数器
                        batch_counters[ds_name] += 1
                        # 清空已保存的结果
                        results_by_dataset[ds_name] = []
    
    # Wait for all processes
    accelerator.wait_for_everyone()
    
    # 每个进程清理自己的临时目录
    process_temp_dir = f'temp_{os.getpid()}'
    if os.path.exists(process_temp_dir) and os.path.isdir(process_temp_dir):
        try:
            # 首先删除目录中的所有文件
            for file in os.listdir(process_temp_dir):
                file_path = os.path.join(process_temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            # 然后删除目录
            os.rmdir(process_temp_dir)
            print(f"Process {process_index} removed its temporary directory: {process_temp_dir}")
        except Exception as e:
            print(f"Process {process_index} failed to remove temporary directory {process_temp_dir}: {e}")
    
    # Main process merges all results
    if is_main_process:
        for dataset_name in set(item['dataset_name'] for item in all_items):
            dataset_dir = next((item['dataset_dir'] for item in all_items if item['dataset_name'] == dataset_name), None)
            if not dataset_dir:
                continue
            
            # Gather all temp files from all processes and all batches
            all_temp_files = []
            # 查找所有可能的临时文件
            for file in os.listdir(dataset_dir):
                if file.startswith(f"{dataset_name}_motion_scores_proc") and file.endswith(".csv"):
                    all_temp_files.append(os.path.join(dataset_dir, file))
            
            if all_temp_files:
                final_csv_path = os.path.join(dataset_dir, f"{dataset_name}_motion_scores.csv")
                print(f"Found {len(all_temp_files)} temp files for {dataset_name}")
                merge_results(all_temp_files, final_csv_path)
                print(f"Merged results for {dataset_name} saved to {final_csv_path}")
    
    accelerator.wait_for_everyone()
    
    if is_main_process:
        print("All processes finished, results merged and saved")
        
        # 最终清理所有临时文件
        print("Final cleanup of any remaining temporary files...")
        total_cleaned = 0
        for dataset_name in set(item['dataset_name'] for item in all_items):
            dataset_dir = next((item['dataset_dir'] for item in all_items if item['dataset_name'] == dataset_name), None)
            if dataset_dir:
                cleaned = cleanup_temp_files(dataset_dir, dataset_name)
                total_cleaned += cleaned
        
        if total_cleaned > 0:
            print(f"Cleaned up a total of {total_cleaned} temporary files")
        else:
            print("No temporary files found for cleanup")
        
        # 添加最终统计
        total_processed = 0
        for dataset_name in set(item['dataset_name'] for item in all_items):
            dataset_dir = next((item['dataset_dir'] for item in all_items if item['dataset_name'] == dataset_name), None)
            if not dataset_dir:
                continue
                
            csv_path = os.path.join(dataset_dir, f"{dataset_name}_motion_scores.csv")
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8', newline='') as f:
                    # 计算CSV中的行数（减去标题行）
                    line_count = sum(1 for _ in f) - 1
                    
                # 计算该数据集中的项目总数
                dataset_total = sum(1 for item in all_items if item['dataset_name'] == dataset_name)
                
                print(f"Dataset {dataset_name}: Processed {line_count} of {dataset_total} videos ({line_count/dataset_total*100:.2f}%)")
                total_processed += line_count
        
        # 计算所有数据集的总计
        all_total = len(all_items)
        print(f"Overall: Processed {total_processed} of {all_total} videos ({total_processed/all_total*100:.2f}%)")
        
        # 清理临时图像目录
        cleaned_dirs = cleanup_temp_directories()
        if cleaned_dirs > 0:
            print(f"Cleaned up {cleaned_dirs} temporary image directories")

if __name__ == '__main__':
    main() 