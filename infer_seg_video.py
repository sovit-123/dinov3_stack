from src.img_seg.model import Dinov3Segmentation
from src.img_seg.utils import (
    draw_segmentation_map, 
    image_overlay,
    get_segment_labels
)
from src.utils.common import get_dinov3_paths

import argparse
import cv2
import time
import os
import torch
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    help='path to the input video',
    default='input/inference_data/videos/video_1.mp4'
)
parser.add_argument(
    '--device',
    default='cuda:0',
    help='compute device, cpu or cuda'
)
parser.add_argument(
    '--imgsz', 
    default=[448, 448],
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--model',
    default='outputs/model_iou.pth'
)
parser.add_argument(
    '--config',
    required=True,
    help='path to the dataset configuration file'
)
parser.add_argument(
    '--repo-dir',
    dest='repo_dir',
    help='path to the cloned DINOv3 repository'
)
parser.add_argument(
    '--model-name',
    dest='model_name',
    help='name of the model, check: https://github.com/facebookresearch/dinov3?tab=readme-ov-file#pretrained-backbones-via-pytorch-hub',
    default='dinov3_vits16'
)
parser.add_argument(
    '--feautre-extractor',
    dest='feature_extractor',
    default='multi',
    choices=['last', 'multi'],
    help='whether to use layer or multiple layers as features'
)
parser.add_argument(
    '--out-dir',
    dest='out_dir',
    default='inference_results_segmentation',
    help='output sub-directory path inside the `outputs` directory'
)
args = parser.parse_args()

DINOV3_REPO, DINOV3_WEIGHTS = get_dinov3_paths()

out_dir = os.path.join('outputs', args.out_dir)
os.makedirs(out_dir, exist_ok=True)

# Set configurations.
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    f.close()

ALL_CLASSES = config['ALL_CLASSES']
VIZ_MAP = config['VIS_LABEL_MAP']

model = Dinov3Segmentation(
    fine_tune=False, 
    weights=None,
    num_classes=len(ALL_CLASSES),
    repo_dir=DINOV3_REPO,
    model_name=args.model_name,
    feature_extractor=args.feature_extractor
    )

ckpt = torch.load(args.model)
model.load_state_dict(ckpt['model_state_dict'])
_ = model.to(args.device).eval()

cap = cv2.VideoCapture(args.input)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
vid_fps = int(cap.get(5))
save_name = args.input.split(os.path.sep)[-1].split('.')[0]

# Define codec and create VideoWriter object.
# out = cv2.VideoWriter(f"{out_dir}/{save_name}.mp4", 
#                     cv2.VideoWriter_fourcc(*'mp4v'), vid_fps, 
#                     (frame_width, frame_height))
out = cv2.VideoWriter(f"{out_dir}/{save_name}.mp4", 
                    cv2.VideoWriter_fourcc(*'mp4v'), vid_fps, 
                    (frame_width, frame_height))

frame_count = 0
total_fps = 0
while cap.isOpened:
    ret, frame = cap.read()

    if ret:
        frame_count += 1

        # Get original frame size.
        imgsz = frame.shape[:2]

        image = frame
        if args.imgsz is not None:
            image = cv2.resize(image, (args.imgsz[0], args.imgsz[1]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        # Get labels.
        start_time = time.time()
        # labels = get_segment_labels(image, model, args.device, args.imgsz)
        labels = get_segment_labels(image, model, args.device, image.shape[0:2])
        end_time = time.time()

        fps = 1 / (end_time - start_time)
        total_fps += fps
        
        # Get segmentation map.
        seg_map = draw_segmentation_map(labels.cpu(), viz_map=VIZ_MAP)
        # Resize segmentation map to original frame size.
        seg_map = cv2.resize(seg_map, imgsz[::-1])

        outputs = image_overlay(frame[..., ::-1], seg_map) # Original frame in RGB format, and seg map.

        cv2.putText(
            outputs,
            f"{fps:.1f} FPS",
            (15, 35),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        out.write(outputs)
        cv2.imshow('Image', outputs)
        # Press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release VideoCapture().
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()

# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")