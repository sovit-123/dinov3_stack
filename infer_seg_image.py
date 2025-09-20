"""
Script to run image inference.

USAGE:
python infer_seg_image.py --input input/inference_data/ \
--model outputs/img_seg/best_model_iou.pth \
--config segmentation_configs/voc.yaml
"""

from src.img_seg.model import Dinov3Segmentation
from src.img_seg.utils import (
    draw_segmentation_map, 
    image_overlay,
    get_segment_labels
)
from src.utils.common import get_dinov3_paths

import argparse
import cv2
import os
import glob
import torch
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    help='path to the input image directory',
    default='input/inference_data/images'
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
    '--feature-extractor',
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

image_paths = glob.glob(os.path.join(args.input, '*'))
for image_path in image_paths:
    orig_image = cv2.imread(image_path)

    # Get original image size (h, w).
    imgsz = orig_image.shape[:2]

    if args.imgsz is not None:
        image = cv2.resize(orig_image, (args.imgsz[0], args.imgsz[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get labels.
    labels = get_segment_labels(image, model, args.device, args.imgsz)
    
    # Get segmentation map.
    seg_map = draw_segmentation_map(labels.cpu(), viz_map=VIZ_MAP)
    # Resize segmentation map to original image size.
    seg_map = cv2.resize(seg_map, imgsz[::-1])

    outputs = image_overlay(orig_image[..., ::-1], seg_map) # Original image in RGB format, and seg map.
    cv2.imshow('Image', outputs)
    cv2.waitKey(0)
    
    # Save path.
    image_name = image_path.split(os.path.sep)[-1]
    save_path = os.path.join(
        out_dir, '_'+image_name
    )
    cv2.imwrite(save_path, outputs)