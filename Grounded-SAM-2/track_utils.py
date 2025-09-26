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
from multiprocessing import shared_memory
import gc

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--p', type=int, default=1)
parser.add_argument('--step', type=int, default=1)
args = parser.parse_args()
parallel = args.p

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 9990+parallel))
server_socket.listen()


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

device_count = torch.cuda.device_count()
if device_count > 1:
    print(f"Found {device_count} GPUs, will distribute models across GPUs")

    for i in range(device_count):
        with torch.cuda.device(f"cuda:{i}"):
            torch.cuda.empty_cache()

    print(f"Memory before loading models:")
    for i in range(device_count):
        mem_info = torch.cuda.mem_get_info(i)
        free_mem = mem_info[0] / 1024**3
        total_mem = mem_info[1] / 1024**3
        print(f"GPU {i}: {free_mem:.2f}GB free / {total_mem:.2f}GB total")

    video_predictor_device_id = 0
    sam2_image_model_device_id = 1 if device_count > 1 else 0
    grounding_model_device_id = 2 if device_count > 2 else 0

    device_map = {
        "video_predictor_device": video_predictor_device_id,
        "sam2_image_model_device": sam2_image_model_device_id,
        "grounding_model_device": grounding_model_device_id
    }
    device = "cuda"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = {
        "video_predictor_device": 0,
        "sam2_image_model_device": 0,
        "grounding_model_device": 0
    }
print("device", device)
print(f"Device map: {device_map}")

video_predictor_device = f"cuda:{device_map['video_predictor_device']}" if device_count > 1 else device
sam2_image_model_device = f"cuda:{device_map['sam2_image_model_device']}" if device_count > 1 else device
grounding_model_device = f"cuda:{device_map['grounding_model_device']}" if device_count > 1 else device

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
# 将sam2_image_model在指定GPU上加载
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=sam2_image_model_device)
image_predictor = SAM2ImagePredictor(sam2_image_model)


# init grounding dino model from huggingface
model_id = "/path/to/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_id,
    device_map=grounding_model_device
)

if device_count > 1:
    print(f"Memory after loading models:")
    for i in range(device_count):
        mem_info = torch.cuda.mem_get_info(i)
        free_mem = mem_info[0] / 1024**3
        total_mem = mem_info[1] / 1024**3
        print(f"GPU {i}: {free_mem:.2f}GB free / {total_mem:.2f}GB total")

print("\033[1;32mGroundingSAM2 model loaded\033[0m")


def get_next_gpu(current_frame_idx):
    if torch.cuda.device_count() <= 1:
        return 0
    return current_frame_idx % torch.cuda.device_count()


def tracking(video_dir, text):
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # init video predictor state
    with torch.cuda.device(video_predictor_device if torch.cuda.device_count() > 1 else device):
        inference_state = video_predictor.init_state(video_path=video_dir, offload_video_to_cpu=True, async_loading_frames=True)

    step = args.step # the step to sample frames for Grounding DINO predictor

    sam2_masks = MaskDictionaryModel()
    PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
    objects_count = 0

    """
    Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
    """
    print("Total frames:", len(frame_names))
    video_segments = {}  # output the following {step} frames tracking masks

    if torch.cuda.device_count() > 1:
        for gpu_id in range(torch.cuda.device_count()):
            with torch.cuda.device(f"cuda:{gpu_id}"):
                torch.cuda.empty_cache()
        gc.collect()

    for start_frame_idx in range(0, len(frame_names), step):
        current_device = grounding_model_device

        # prompt grounding dino to get the box coordinates on specific frame
        print(f"start_frame_idx {start_frame_idx}, using device {current_device}")

        img_path = os.path.join(video_dir, frame_names[start_frame_idx])
        image = Image.open(img_path)
        image_base_name = frame_names[start_frame_idx].split(".")[0]
        mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

        if torch.cuda.device_count() > 1:
            with torch.cuda.device(current_device):
                torch.cuda.empty_cache()

        # run Grounding DINO on the image
        try:
            inputs = processor(images=image, text=text, return_tensors="pt").to(current_device)

            with torch.no_grad():
                outputs = grounding_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.25,
                text_threshold=0.25,
                target_sizes=[image.size[::-1]]
            )
        except RuntimeError as e:
            print(f"Error processing frame {start_frame_idx} on device {current_device}: {e}")
            print("Falling back to CPU processing for this frame")
            inputs = processor(images=image, text=text, return_tensors="pt").to("cpu")
            with torch.no_grad():
                grounding_model_cpu = grounding_model.to("cpu")
                outputs = grounding_model_cpu(**inputs)
                grounding_model.to(current_device)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.25,
                text_threshold=0.25,
                target_sizes=[image.size[::-1]]
            )

        with torch.no_grad():
            # prompt SAM image predictor to get the mask for the object
            try:
                image_predictor.set_image(np.array(image.convert("RGB")))
            except RuntimeError as e:
                print(f"Error setting image on device {sam2_image_model_device}: {e}")
                print("Falling back to CPU for image processing")
                image_predictor.model.to("cpu")
                image_predictor.set_image(np.array(image.convert("RGB")))
                image_predictor.model.to(sam2_image_model_device)

            # process the detection results
            input_boxes = results[0]["boxes"] # .cpu().numpy()
            # print("results[0]",results[0])
            OBJECTS = results[0]["labels"]
            if input_boxes.shape[0] != 0:
                # prompt SAM 2 image predictor to get the mask for the object
                masks, scores, logits = image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
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

                # If you are using point prompts, we uniformly sample positive points based on the mask
                if mask_dict.promote_type == "mask":
                    target_device = sam2_image_model_device

                    try:
                        mask_tensor = torch.tensor(masks).to(target_device)
                        boxes_tensor = torch.tensor(input_boxes).to(target_device)
                        mask_dict.add_new_frame_annotation(mask_list=mask_tensor, box_list=boxes_tensor, label_list=OBJECTS)
                    except RuntimeError as e:
                        print(f"Error moving tensors to device {target_device}: {e}")
                        mask_tensor = torch.tensor(masks).cpu()
                        boxes_tensor = torch.tensor(input_boxes).cpu()
                        mask_dict.add_new_frame_annotation(mask_list=mask_tensor, box_list=boxes_tensor, label_list=OBJECTS)
                else:
                    raise NotImplementedError("SAM 2 video predictor only support mask prompts")


                """
                Step 4: Propagate the video predictor to get the segmentation results for each frame
                """
                objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
                print("objects_count", objects_count)
            else:
                print("No object detected in the frame, skip merge the frame merge {}".format(frame_names[start_frame_idx]))
                mask_dict = sam2_masks


            if len(mask_dict.labels) == 0:
                print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
                continue
            else:
                with torch.cuda.device(video_predictor_device if torch.cuda.device_count() > 1 else device):
                    video_predictor.reset_state(inference_state)

                    for object_id, object_info in mask_dict.labels.items():
                        try:
                            if torch.cuda.device_count() > 1:
                                object_mask = object_info.mask.to(video_predictor_device)
                            else:
                                object_mask = object_info.mask

                            frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                                    inference_state,
                                    start_frame_idx,
                                    object_id,
                                    object_mask,
                                )
                        except RuntimeError as e:
                            print(f"Error adding mask: {e}, trying CPU fallback")
                            # CPU fallback
                            object_mask = object_info.mask.cpu()
                            frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                                    inference_state,
                                    start_frame_idx,
                                    object_id,
                                    object_mask,
                                )

                    video_masks = {}
                    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
                        frame_masks = MaskDictionaryModel()

                        for i, out_obj_id in enumerate(out_obj_ids):
                            out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                            object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id))
                            object_info.update_box()
                            frame_masks.labels[out_obj_id] = object_info
                            image_base_name = frame_names[out_frame_idx].split(".")[0]
                            frame_masks.mask_name = f"mask_{image_base_name}.npy"
                            frame_masks.mask_height = out_mask.shape[-2]
                            frame_masks.mask_width = out_mask.shape[-1]

                        video_masks[out_frame_idx] = frame_masks
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                        sam2_masks = copy.deepcopy(frame_masks)

                    print("video_segments:", len(video_segments))

        if torch.cuda.device_count() > 1:
            del outputs, inputs
            if 'mask_tensor' in locals():
                del mask_tensor
            if 'boxes_tensor' in locals():
                del boxes_tensor
            for gpu_id in range(torch.cuda.device_count()):
                with torch.cuda.device(f"cuda:{gpu_id}"):
                    torch.cuda.empty_cache()
            gc.collect()

    """
    Step 5: Visualize the segment results across the video and save them
    """
    save_dir = "../tracking_results"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # # ID_TO_OBJECTS mapping
    # ID_TO_OBJECTS = {}
    # for frame_masks in video_masks.values():
    #     for obj_id, obj_info in frame_masks.labels.items():
    #         if obj_id not in ID_TO_OBJECTS:
    #             ID_TO_OBJECTS[obj_id] = obj_info.class_name

    print(f"Single Frame's shape: {image.size}")

    # Dictionary to store detection results
    detection_results = {}

    for frame_idx, segments in video_segments.items():
        img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))

        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
            mask=masks, # (n, h, w)
            class_id=np.array(object_ids, dtype=np.int32),
        )

        # Store detection boxes for each object in this frame
        frame_detections = {}
        for i, obj_id in enumerate(object_ids):
            # Convert to list for serialization
            box = detections.xyxy[i].tolist()
            # obj_name = ID_TO_OBJECTS[obj_id]
            frame_detections[obj_id] = box

        # Add to results dictionary
        detection_results[frame_idx] = frame_detections

        # Visualize and save annotated frames
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections)
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

        # Print detection information
        print(f"Frame {frame_idx} detections: {len(detections.xyxy)} objects:")
        for i in range(len(detections.xyxy)):
            print(f"  {i}. {detections.xyxy[i]}")

    if torch.cuda.device_count() > 1:
        for gpu_id in range(torch.cuda.device_count()):
            with torch.cuda.device(f"cuda:{gpu_id}"):
                torch.cuda.empty_cache()
        gc.collect()

    # Return the detection results dictionary
    return detection_results, video_segments, None

def main():
    print("Tracking agent is listening...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        data = client_socket.recv(8192)
        if not data:
            break

        # Deserialize data
        frames_path, obj_list = pickle.loads(data)

        prompt = ""
        if obj_list is not None:
            for index, obj in enumerate(obj_list):
                # Process objects from list 1
                prompt += f"{obj}. "
        result = tracking(frames_path, prompt[:-1])

        # Check if result is None (no objects detected)
        if result is None:
            result_data = pickle.dumps(None)
        else:
            result_data = pickle.dumps(result)

        # Get access to existing shared memory
        try:
            shm = shared_memory.SharedMemory(name=f"MySharedMemory{parallel if parallel>1 else ''}",
                create=True,
                size=8+len(result_data)
            )
            size = len(result_data)
            size_bytes = size.to_bytes(8, byteorder='big')
            shm.buf[0:8] = size_bytes
            shm.buf[8:8+size] = result_data
            client_socket.send(b'done')
            shm.close()

        except Exception as e:
            print(f"Shared memory error: {e}")
            # Send error to client
            client_socket.send(b'error')

if __name__ == "__main__":
    main()
