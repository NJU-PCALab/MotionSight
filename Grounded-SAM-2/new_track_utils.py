import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import hydra
import json
import copy
import socket
import pickle
import gc
import time
print("\033[1;33mLoading GroundingSAM2 model...\033[0m")
"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# hydra is initialized on import of sam2, which sets the search path which can't be modified
# so we need to clear the hydra instance
hydra.core.global_hydra.GlobalHydra.instance().clear()
# reinit hydra with a new search path for configs
hydra.initialize_config_module('sam2', version_base='1.2')

device = "cuda"

# init sam image predictor and video predictor model
sam2_checkpoint = "/path/to/Grounded_SAM_2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
# 将sam2_image_model在指定GPU上加载
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)


# init grounding dino model from huggingface
model_id = "/path/to/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
# 明确指定grounding模型在单一GPU上，不使用自动分配
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

print("\033[1;32mGroundingSAM2 model loaded\033[0m")



def tracking(video_dir, text, step=10000, return_object_names=False):
    total_start_time = time.time()

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # init video predictor state
    init_state_time = time.time()
    inference_state = video_predictor.init_state(video_path=video_dir, offload_video_to_cpu=True, async_loading_frames=True)
    print(f"Initialize state time: {time.time() - init_state_time:.2f}s")

    sam2_masks = MaskDictionaryModel()
    PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
    objects_count = 0

    """
    Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
    """
    print("Total frames:", len(frame_names))
    video_segments = {}  # output the following {step} frames tracking masks

    # 在多GPU环境中，优化内存管理
    if torch.cuda.device_count() > 1:
        # 清理所有GPU缓存，确保每次处理前有足够的显存
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        # 强制垃圾回收
        gc.collect()

    for start_frame_idx in range(0, len(frame_names), step):
        frame_start_time = time.time()

        img_path = os.path.join(video_dir, frame_names[start_frame_idx])
        image = Image.open(img_path)
        image_base_name = frame_names[start_frame_idx].split(".")[0]
        mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

        # 在多GPU环境中，确保处理每个批次前清理缓存
        if torch.cuda.device_count() > 1:
            # 清理当前GPU的缓存
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

        # run Grounding DINO on the image
        grounding_start_time = time.time()
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )
        print(f"Grounding DINO time: {time.time() - grounding_start_time:.2f}s")

        # prompt SAM image predictor to get the mask for the object
        sam_start_time = time.time()
        image_predictor.set_image(np.array(image.convert("RGB")))

        # process the detection results
        input_boxes = results[0]["boxes"]
        OBJECTS = results[0]["labels"]
        if input_boxes.shape[0] != 0:
            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            print(f"SAM prediction time: {time.time() - sam_start_time:.2f}s")

            # convert the mask shape to (n, H, W)
            if masks.ndim == 2:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)

            """
            Step 3: Register each object's positive points to video predictor
            """
            register_start_time = time.time()
            if mask_dict.promote_type == "mask":
                mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
            else:
                raise NotImplementedError("SAM 2 video predictor only support mask prompts")
            print(f"Mask registration time: {time.time() - register_start_time:.2f}s")

            """
            Step 4: Propagate the video predictor to get the segmentation results for each frame
            """
            update_start_time = time.time()
            objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
            print(f"Mask update time: {time.time() - update_start_time:.2f}s")
            print("objects_count", objects_count)
        else:
            print("No object detected in the frame, skip merge the frame merge {}".format(frame_names[start_frame_idx]))
            mask_dict = sam2_masks

        if len(mask_dict.labels) == 0:
            print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
            continue
        else:
            propagate_start_time = time.time()
            video_predictor.reset_state(inference_state)

            for object_id, object_info in mask_dict.labels.items():
                frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                        inference_state,
                        start_frame_idx,
                        object_id,
                        object_info.mask,
                    )

            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
                frame_masks = MaskDictionaryModel()

                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0)
                    object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id))
                    object_info.update_box()
                    frame_masks.labels[out_obj_id] = object_info
                    image_base_name = frame_names[out_frame_idx].split(".")[0]
                    frame_masks.mask_name = f"mask_{image_base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                sam2_masks = copy.deepcopy(frame_masks)

            print(f"Propagation time: {time.time() - propagate_start_time:.2f}s")
            print("video_segments:", len(video_segments))

        print(f"Total frame processing time: {time.time() - frame_start_time:.2f}s")

        # 如果存在多GPU，周期性清理GPU内存
        if torch.cuda.device_count() > 1:
            # 释放不需要的显存
            del outputs, inputs
            if 'mask_tensor' in locals():
                del mask_tensor
            if 'boxes_tensor' in locals():
                del boxes_tensor
            # 清理所有GPU
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            # 强制垃圾回收
            gc.collect()

    """
    Step 5: Visualize the segment results across the video and save them
    """
    visualization_start_time = time.time()

    print(f"Single Frame's shape: {image.size}")

    # Dictionary to store detection results
    detection_results = {}

    for frame_idx, segments in video_segments.items():
        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
            mask=masks, # (n, h, w)
            class_id=np.array(object_ids, dtype=np.int32),
        )

        frame_detections = {}
        for i, obj_id in enumerate(object_ids):
            box = detections.xyxy[i].tolist()
            if return_object_names:
                class_name = sam2_masks.get_target_class_name(obj_id)
                frame_detections[obj_id] = {"box": box, "class_name": class_name}
            else:
                frame_detections[obj_id] = box

        detection_results[frame_idx] = frame_detections

        print(f"Frame {frame_idx} detections: {len(detections.xyxy)} objects:")

    print(f"Visualization time: {time.time() - visualization_start_time:.2f}s")
    print(f"Total processing time: {time.time() - total_start_time:.2f}s")

    # 最后一次清理显存，确保资源被释放
    if torch.cuda.device_count() > 1:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        # 强制垃圾回收
        gc.collect()

    return detection_results

def run_track(frames_path, obj_list, return_object_names=False):
    prompt = ""
    if obj_list is not None:
        for index, obj in enumerate(obj_list):
            # Process objects from list 1
            prompt += f"{obj}. "
    result = tracking(frames_path, prompt[:-1], return_object_names=return_object_names)

    return result

# if __name__ == "__main__":
#     main()
