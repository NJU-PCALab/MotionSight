import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image

siglip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=torch.float16, device_map="auto")
siglip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
device = "cuda"

# TODO: maybe have some bugs
def select_boxes(video, boxes, text, m=1):
    num_frames, height, width, _ = video.shape
    text_prompt = f"A picture of {text}"

    num_boxes = len(boxes)

    if num_boxes == 0:
        return []

    text_inputs = siglip_processor(text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = siglip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    box_scores = torch.zeros(num_boxes, device=device)
    for box_idx, box in enumerate(boxes):
        for frame_idx in range(num_frames):
            frame = video[frame_idx]
            cx, cy, w, h = box * torch.Tensor([width, height, width, height])
            x1, y1 = int(cx - w/2), int(cy - h/2)
            x2, y2 = int(cx + w/2), int(cy + h/2)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            cropped_region = frame[y1:y2, x1:x2, :]

            if cropped_region.size == 0 or x2 <= x1 or y2 <= y1:
                print(f"Box is invalid, skipping...")
                continue

            # Convert numpy array to PIL Image
            cropped_pil = Image.fromarray(cropped_region)
            image_inputs = siglip_processor(images=cropped_pil, return_tensors="pt").to(device)

            with torch.no_grad():
                image_features = siglip_model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            if image_features.device != text_features.device:
                text_features = text_features.to(image_features.device)

            similarity = (image_features @ text_features.T).item()
            box_scores[box_idx] += similarity



    if m > num_boxes:
        m = num_boxes

    top_indices = torch.topk(box_scores, m).indices.cpu().numpy()
    top_boxes = [boxes[idx] for idx in top_indices]

    return top_boxes
