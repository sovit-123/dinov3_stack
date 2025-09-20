"""
Script to analyze and visualze anchors.

Execute from parent project directory as:
python -m src.detection.analyze_anchors
"""

import argparse
import yaml
import cv2
import matplotlib.pyplot as plt
import torch
import os
import random
import numpy as np
import torch.nn as nn
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.image_list import ImageList

# ----------------------
# Load dataset config
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    default='detection_configs/voc.yaml',
    help='path to the dataset configuration file'
)
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

CLASSES = config['CLASSES']
TRAIN_IMG, TRAIN_ANNOT = config['TRAIN_IMG'], config['TRAIN_ANNOT']
VALID_IMG, VALID_ANNOT = config['VALID_IMG'], config['VALID_ANNOT']

# ----------------------
# Load a sample image
# ----------------------
image_files = [f for f in os.listdir(TRAIN_IMG) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    raise FileNotFoundError(f"No images found in {TRAIN_IMG}")

# Pick a random sample
sample_image_path = os.path.join(TRAIN_IMG, random.choice(image_files))

print(f"Visualizing anchors on: {sample_image_path}")
image = cv2.imread(sample_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize to SSD input size
resize_h, resize_w = 640, 640
resized = cv2.resize(image, (resize_w, resize_h))

# Convert to tensor and normalize for torchvision
image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

# Create ImageList object instead of using tuple
image_list = ImageList(image_tensor, [(resize_h, resize_w)])

# ----------------------
# Create mock feature maps for demonstration
# ----------------------
def create_mock_feature_maps():
    """Create mock feature maps with proper shapes"""
    feature_maps = []
    # Simulate feature maps from different layers
    for size in [(80, 80), (40, 40), (20, 20), (10, 10), (5, 5), (3, 3)]:
        # Create a mock feature map with random data
        feat_map = torch.randn(1, 256, size[0], size[1])  # batch=1, channels=256, height, width
        feature_maps.append(feat_map)
    return feature_maps

# Create mock feature maps
feature_maps = create_mock_feature_maps()

# ----------------------
# Build SSD anchor generator - CORRECTED
# ----------------------
# Aspect ratios for each feature map (based on SSD paper)
aspect_ratios = [
    [2.0],           # for 80x80 feature map
    [2.0, 3.0],      # for 40x40 feature map  
    [2.0, 3.0],      # for 20x20 feature map
    [2.0, 3.0],      # for 10x10 feature map
    [2.0],           # for 5x5 feature map
    [2.0]            # for 3x3 feature map
]

# Create DefaultBoxGenerator with correct parameters
anchor_generator = DefaultBoxGenerator(
    aspect_ratios=aspect_ratios,
    min_ratio=0.1,   # Minimum scale ratio
    max_ratio=0.9    # Maximum scale ratio
)

# Generate anchors
anchors_result = anchor_generator(image_list, feature_maps)

print(f"Type of anchor result: {type(anchors_result)}")
if isinstance(anchors_result, list):
    print(f"Length of anchor result list: {len(anchors_result)}")
    # DefaultBoxGenerator returns [single_tensor_with_all_anchors]
    all_anchors = anchors_result[0]  # Get the single tensor
    print(f"All anchors shape: {all_anchors.shape}")
    
    # Now we need to split this properly
    # The DefaultBoxGenerator documentation shows it returns anchors in order:
    # feature_map_0_anchors, feature_map_1_anchors, ..., feature_map_n_anchors
    
    # Calculate how many anchors each feature map should have
    anchors_per_fm = []
    for i, (h, w) in enumerate([(80, 80), (40, 40), (20, 20), (10, 10), (5, 5), (3, 3)]):
        # For DefaultBoxGenerator:
        # - Creates anchors based on min_ratio, max_ratio, and aspect_ratios
        # - Number of scales = 2 (base + extra)
        # - Total per location = num_scales + len(aspect_ratios)
        num_aspect_ratios = len(aspect_ratios[i])
        # DefaultBoxGenerator creates: 2 scales + aspect_ratio_variants
        # But this is complex, let's reverse engineer from total count
        pass
    
    # Since we know total is 38,336, let's try a different approach
    # Let's assume roughly equal distribution weighted by feature map size
    total_anchors = len(all_anchors)
    total_locations = sum(h * w for h, w in [(80, 80), (40, 40), (20, 20), (10, 10), (5, 5), (3, 3)])
    
    print(f"Total locations across all feature maps: {total_locations}")
    print(f"Average anchors per location: {total_anchors / total_locations:.2f}")
    
    # Calculate expected anchors per feature map based on aspect ratios
    expected_anchors = []
    for i, (h, w) in enumerate([(80, 80), (40, 40), (20, 20), (10, 10), (5, 5), (3, 3)]):
        # DefaultBoxGenerator formula: for each location
        # - 1 base anchor
        # - 1 additional scale anchor
        # - len(aspect_ratios) aspect ratio variants (excluding 1.0 which is the base)
        num_ar = len(aspect_ratios[i])
        anchors_per_location = 2 + num_ar  # 2 scales + aspect ratios
        total_for_fm = h * w * anchors_per_location
        expected_anchors.append(total_for_fm)
        print(f"FM {i} ({h}x{w}): {anchors_per_location} anchors/location = {total_for_fm} expected")
    
    expected_total = sum(expected_anchors)
    print(f"Expected total: {expected_total}, Actual total: {total_anchors}")
    
    if expected_total != total_anchors:
        # If our calculation is wrong, let's try empirical splitting
        print("Using empirical splitting based on feature map sizes...")
        fm_sizes = [h * w for h, w in [(80, 80), (40, 40), (20, 20), (10, 10), (5, 5), (3, 3)]]
        total_size = sum(fm_sizes)
        
        expected_anchors = []
        for fm_size in fm_sizes:
            proportion = fm_size / total_size
            fm_anchor_count = int(total_anchors * proportion)
            expected_anchors.append(fm_anchor_count)
    
    # Split the tensor
    anchors_over_all_feature_maps = []
    start_idx = 0
    for i, expected_count in enumerate(expected_anchors):
        if i == len(expected_anchors) - 1:  # Last feature map gets remainder
            end_idx = len(all_anchors)
        else:
            end_idx = start_idx + expected_count
        
        fm_anchors = all_anchors[start_idx:end_idx]
        anchors_over_all_feature_maps.append(fm_anchors)
        print(f"FM {i}: {len(fm_anchors)} anchors (indices {start_idx}:{end_idx})")
        start_idx = end_idx

else:
    print("Unexpected anchor format")
    anchors_over_all_feature_maps = []

print(f"\nFinal result:")
print(f"Number of feature maps: {len(anchors_over_all_feature_maps)}")
for i, anchors in enumerate(anchors_over_all_feature_maps):
    print(f"Feature map {i}, anchors shape: {anchors.shape}")

# Convert anchors to numpy for easier manipulation
anchors_numpy = [anchors.cpu().numpy() for anchors in anchors_over_all_feature_maps]

# ----------------------
# Debug: Check anchor ranges
# ----------------------
print("\n" + "="*50)
print("DEBUG: ANCHOR RANGES")
print("="*50)

for i, anchors in enumerate(anchors_numpy):
    if len(anchors) > 0:
        x1_min, y1_min = anchors[:, 0].min(), anchors[:, 1].min()
        x2_max, y2_max = anchors[:, 2].max(), anchors[:, 3].max()
        print(f"FM {i}: x1_min={x1_min:.1f}, y1_min={y1_min:.1f}, x2_max={x2_max:.1f}, y2_max={y2_max:.1f}")
        
        # Count valid anchors (within image bounds)
        valid_mask = (anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & \
                    (anchors[:, 2] <= resize_w) & (anchors[:, 3] <= resize_h) & \
                    (anchors[:, 2] > anchors[:, 0]) & (anchors[:, 3] > anchors[:, 1])
        valid_count = valid_mask.sum()
        print(f"FM {i}: {valid_count}/{len(anchors)} anchors are valid and within bounds")

# ----------------------
# Visualize anchors on 640x640 image but keep the insightful analysis
# ----------------------
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

# Get the actual feature map shapes from the mock feature maps
feature_map_shapes = [feat_map.shape[-2:] for feat_map in feature_maps]

for idx in range(min(6, len(anchors_numpy))):
    ax = axes[idx]
    
    # Show full 640x640 image in each subplot
    ax.imshow(resized)
    
    anchors = anchors_numpy[idx]
    h, w = feature_map_shapes[idx]
    
    print(f"\nProcessing Feature Map {idx}: {h}x{w}, total anchors: {len(anchors)}")
    
    if len(anchors) == 0:
        print(f"Warning: No anchors found for feature map {idx}")
        ax.set_title(f"FM {idx} ({h}x{w})\nNo anchors found")
        continue
    
    # Filter valid anchors first
    valid_mask = (anchors[:, 0] >= -50) & (anchors[:, 1] >= -50) & \
                (anchors[:, 2] <= resize_w + 50) & (anchors[:, 3] <= resize_h + 50) & \
                (anchors[:, 2] > anchors[:, 0]) & (anchors[:, 3] > anchors[:, 1])
    
    valid_anchors = anchors[valid_mask]
    print(f"Valid anchors for FM {idx}: {len(valid_anchors)}/{len(anchors)}")
    
    if len(valid_anchors) == 0:
        print(f"Warning: No valid anchors for feature map {idx}")
        ax.set_title(f"FM {idx} ({h}x{w})\nNo valid anchors")
        continue
    
    # Sample anchors for visualization (don't make it too dense)
    max_anchors_to_show = min(200, len(valid_anchors))  # Show at most 200 anchors
    if len(valid_anchors) > max_anchors_to_show:
        # Sample uniformly
        indices = np.linspace(0, len(valid_anchors)-1, max_anchors_to_show, dtype=int)
        sampled_anchors = valid_anchors[indices]
    else:
        sampled_anchors = valid_anchors
    
    print(f"Showing {len(sampled_anchors)} anchors for FM {idx}")
    
    # Calculate scale factor for reference
    scale_x = w / resize_w
    scale_y = h / resize_h
    
    # Draw sampled anchors on full 640x640 image
    for anchor in sampled_anchors:
        x1, y1, x2, y2 = anchor
        
        # Clamp coordinates to reasonable bounds
        x1, y1 = max(-10, x1), max(-10, y1)
        x2, y2 = min(resize_w + 10, x2), min(resize_h + 10, y2)
        
        # Only draw if it's a reasonable box
        if x2 > x1 and y2 > y1 and (x2-x1) > 2 and (y2-y1) > 2:
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="red", linewidth=0.8, alpha=0.7
            )
            ax.add_patch(rect)
    
    ax.set_title(f"FM {idx} ({h}x{w})\n{len(anchors)} total, {len(sampled_anchors)} shown\nScale: {scale_x:.3f}x")
    ax.set_xlim(0, resize_w)
    ax.set_ylim(resize_h, 0)

plt.tight_layout()
plt.show()

# ----------------------
# Additional: Show anchor size distribution per feature map
# ----------------------
print("\n" + "="*70)
print("ANCHOR SIZE ANALYSIS AT EACH FEATURE MAP SCALE")
print("="*70)

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
axes2 = axes2.flatten()

for idx in range(min(6, len(anchors_numpy))):
    ax = axes2[idx]
    
    anchors = anchors_numpy[idx]
    h, w = feature_map_shapes[idx]
    
    if len(anchors) == 0:
        continue
        
    # Calculate anchor areas and aspect ratios in original coordinate system
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    areas = widths * heights
    aspect_ratios_computed = widths / np.maximum(heights, 1e-6)
    
    # Scale to feature map coordinates for better understanding
    scale_x = w / resize_w
    scale_y = h / resize_h
    scaled_widths = widths * scale_x
    scaled_heights = heights * scale_y
    scaled_areas = scaled_widths * scaled_heights
    
    # Plot area distribution
    ax.hist(scaled_areas, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel(f'Anchor Area (in {h}x{w} coordinates)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'FM {idx} ({h}x{w})\nAnchor Area Distribution\nMean: {scaled_areas.mean():.2f}')
    ax.grid(True, alpha=0.3)
    
    print(f"\nFeature Map {idx} ({h}x{w}):")
    print(f"  Original coord - Width: {widths.min():.1f}-{widths.max():.1f}, Height: {heights.min():.1f}-{heights.max():.1f}")
    print(f"  Scaled coord   - Width: {scaled_widths.min():.1f}-{scaled_widths.max():.1f}, Height: {scaled_heights.min():.1f}-{scaled_heights.max():.1f}")
    print(f"  Scaled areas: {scaled_areas.min():.1f} - {scaled_areas.max():.1f} (mean: {scaled_areas.mean():.1f})")
    print(f"  Aspect ratios: {aspect_ratios_computed.min():.2f} - {aspect_ratios_computed.max():.2f}")

plt.tight_layout()
plt.show()

# ----------------------
# Additional Statistics
# ----------------------
print("\n" + "="*50)
print("DETAILED ANCHOR STATISTICS")
print("="*50)

total_anchors = sum(len(anchors) for anchors in anchors_numpy)
print(f"Total number of anchors: {total_anchors}")

for i, anchors in enumerate(anchors_numpy):
    if len(anchors) == 0:
        print(f"\nFeature Map {i} ({feature_map_shapes[i]}): NO ANCHORS!")
        continue
        
    areas = []
    aspect_ratios_computed = []
    widths = []
    heights = []
    
    for anchor in anchors:
        x1, y1, x2, y2 = anchor
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        areas.append(area)
        aspect_ratios_computed.append(aspect_ratio)
        widths.append(width)
        heights.append(height)
    
    print(f"\nFeature Map {i} ({feature_map_shapes[i]}):")
    print(f"  Number of anchors: {len(anchors)}")
    print(f"  Width range: {min(widths):.1f} - {max(widths):.1f} px")
    print(f"  Height range: {min(heights):.1f} - {max(heights):.1f} px")
    print(f"  Area range: {min(areas):.1f} - {max(areas):.1f} px²")
    print(f"  Aspect ratio range: {min(aspect_ratios_computed):.2f} - {max(aspect_ratios_computed):.2f}")
    
    # Count anchors within image bounds
    within_bounds = 0
    for anchor in anchors:
        x1, y1, x2, y2 = anchor
        if (0 <= x1 < resize_w and 0 <= x2 <= resize_w and
            0 <= y1 < resize_h and 0 <= y2 <= resize_h and
            x2 > x1 and y2 > y1):
            within_bounds += 1
    
    coverage = (within_bounds / len(anchors)) * 100
    print(f"  Coverage: {coverage:.2f}% within image bounds ({within_bounds}/{len(anchors)})")