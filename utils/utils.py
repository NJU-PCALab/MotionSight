import json
import numpy as np
import torch.nn.functional as F
import os
# import matplotlib
# import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from pprint import pformat

def setup_logging(name="main_agent.log", log_dir="logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    backup_dir = os.path.join(log_dir, "backups")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    log_file = os.path.join(log_dir, name)

    if os.path.exists(log_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(name)[0]
        backup_file = os.path.join(backup_dir, f"{base_name}_{timestamp}.log")
        shutil.copy2(log_file, backup_file)
        os.remove(log_file)

    return log_file

def log_to_file(log_file, message, level="INFO", auto_format=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if auto_format and isinstance(message, (dict, list, tuple, set)):
        formatted_msg = pformat(message, indent=2, width=80)
    else:
        formatted_msg = str(message)

    log_entry = f"[{timestamp}] [{level}] {formatted_msg}\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def parser_json(out):
    if "```json" in out:
        # Extract JSON content between ```json and ``` markers
        json_start = out.find("```json") + 7
        json_end = out.find("```", json_start)
        json_content = out[json_start:json_end].strip()
        response_json = json.loads(json_content)
    else:
        response_json = json.loads(out)
    return response_json

# def read_jsonl(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 data.append(json.loads(line.strip()))
#             except json.JSONDecodeError:
#                 logger.error(f"无法解析行: {line}")
#     return data

# def preprocess_frames(frames):
#     """Preprocess frames to model inputs.

#     Args:
#         frames: [num_frames, height, width, 3], [0, 255], np.uint8

#     Returns:
#         frames: [num_frames, height, width, 3], [-1, 1], np.float32
#     """
#     frames = frames.float()
#     frames = frames / 255 * 2 - 1
#     return frames


# def postprocess_occlusions(occlusions, expected_dist):
#     visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
#     return visibles

# # Utility Functions
# def inference(frames, query_points, model):
#     # Preprocess video to match model inputs format
#     frames = preprocess_frames(frames)
#     query_points = query_points.float()
#     frames, query_points = frames[None], query_points[None]

#     # Model inference
#     outputs = model(frames, query_points)
#     tracks, occlusions, expected_dist = (
#         outputs['tracks'][0],
#         outputs['occlusion'][0],
#         outputs['expected_dist'][0],
#     )

#     # Binarize occlusions
#     visibles = postprocess_occlusions(occlusions, expected_dist)
#     return tracks, visibles

# def sample_grid_points(frame_idx, height, width, stride=1):
#     """Sample grid points with (time height, width) order."""
#     points = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
#     points = points.transpose(1, 2, 0)
#     out_height, out_width = points.shape[0:2]
#     frame_idx = np.ones((out_height, out_width, 1)) * frame_idx
#     points = np.concatenate((frame_idx, points), axis=-1).astype(np.int32)
#     points = points.reshape(-1, 3)  # [out_height*out_width, 3]
#     return points

# def plot_tracks_tails(
#     rgb, points, occluded, point_size=12, linewidth=1.5
# ):
#   """Plot rainbow tracks with matplotlib.

#   Points nearby in the points array will be assigned similar colors.
#   Draws the motion trajectory of each point across frames.

#   Args:
#     rgb: rgb pixels of shape [num_frames, height, width, 3], float or uint8.
#     points: Points array, float32, of shape [num_points, num_frames, 2] in x,y
#       order in raster coordinates.
#     occluded: Array of occlusion values, where 1 is occluded and 0 is not, of
#       shape [num_points, num_frames].
#     point_size: to control the scale of the points. Passed to plt.scatter.
#     linewidth: to control the line thickness. Passed to matplotlib LineCollection.

#   Returns:
#     frames: rgb frames with rendered rainbow tracks.
#   """

#   disp = []
#   cmap = plt.cm.hsv  # pytype: disable=module-attr

#   z_list = np.arange(points.shape[0])
#   colors = cmap(z_list / (np.max(z_list) + 1))
#   figure_dpi = 64

#   figs = []
#   for i in range(rgb.shape[0]):
#     print(f'Plotting frame {i}...')
#     fig = plt.figure(
#         figsize=(rgb.shape[2] / figure_dpi, rgb.shape[1] / figure_dpi),
#         dpi=figure_dpi,
#         frameon=False,
#         facecolor='w',
#     )
#     figs.append(fig)
#     ax = fig.add_subplot()
#     ax.axis('off')
#     ax.imshow(rgb[i] / 255.0)

#     # Set axis limits to prevent arrows from expanding the figure
#     ax.set_xlim(0, rgb.shape[2])
#     ax.set_ylim(rgb.shape[1], 0)  # Inverted y-axis for image coordinates

#     # Draw current points as arrows
#     colalpha = np.concatenate(
#         [colors[:, :-1], 1 - occluded[:, i : i + 1]], axis=1
#     )
#     points_clipped = np.maximum(points, 0.0)
#     points_clipped = np.minimum(points_clipped, [rgb.shape[2], rgb.shape[1]])

#     # Instead of scatter points, draw arrows
#     if i > 0:
#       for j in range(points_clipped.shape[0]):
#         if occluded[j, i] < 0.5:  # Only draw visible points
#           # Get current and previous point to determine arrow direction
#           curr_point = points_clipped[j, i]
#           prev_point = points_clipped[j, max(0, i-1)]

#           # Calculate arrow direction
#           dx = curr_point[0] - prev_point[0]
#           dy = curr_point[1] - prev_point[1]

#           # Check if arrow would go outside the frame
#           if (0 <= curr_point[0] <= rgb.shape[2] and
#               0 <= curr_point[1] <= rgb.shape[1] and
#               0 <= curr_point[0] - dx*0.5 <= rgb.shape[2] and
#               0 <= curr_point[1] - dy*0.5 <= rgb.shape[1]):
#             # Draw arrow
#             plt.arrow(
#                 curr_point[0] - dx*0.5, curr_point[1] - dy*0.5, dx*0.5, dy*0.5,
#                 head_width=point_size/3, head_length=point_size/2,
#                 fc=colalpha[j], ec=colalpha[j], linewidth=linewidth
#             )
#     else:
#       # For the first frame, use scatter since we don't have previous points
#       visible_points = occluded[:, i] < 0.5
#       plt.scatter(
#           points_clipped[visible_points, i, 0],
#           points_clipped[visible_points, i, 1],
#           s=point_size,
#           c=colalpha[visible_points]
#       )

#     # Draw trajectory tails (previous positions)
#     for j in range(i - 1, max(0, i - 20), -1):  # Only show last 10 frames for cleaner visualization
#       # Create line segments between consecutive points
#       pts = np.stack([points_clipped[:, j], points_clipped[:, j+1]], axis=1)

#       # Calculate visibility for line segments
#       visibility = (1 - occluded[:, j:j+1]) * (1 - occluded[:, j+1:j+2])

#       # Set color with alpha based on visibility
#       colalpha2 = np.concatenate([colors[:, :-1], visibility], axis=1)

#       # Add line collection
#       plt.gca().add_collection(
#           matplotlib.collections.LineCollection(
#               pts, color=colalpha2, linewidth=linewidth
#           )
#       )

#     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#     plt.margins(0, 0)
#     fig.canvas.draw()
#     width, height = fig.get_size_inches() * fig.get_dpi()
#     img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
#         int(height), int(width), 3
#     )
#     disp.append(np.copy(img))

#   for fig in figs:
#     plt.close(fig)
#   return np.stack(disp, axis=0)