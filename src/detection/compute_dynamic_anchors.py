import numpy as np
import xml.etree.ElementTree as ET
import os
import cv2
import yaml
import matplotlib.pyplot as plt
import argparse

from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    default='detection_configs/voc.yaml',
    help='path to the dataset configuration file'
)
parser.add_argument(
    '--max-samples',
    default=1000,
    type=int,
    dest='max_samples',
    help='maximum number of images to use to compute anchor aspect ratios'
)
args = parser.parse_args()

def analyze_dataset_bboxes(config_path, max_samples=1000):
    """
    Analyze bounding boxes in your dataset to determine optimal anchor configurations
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    TRAIN_IMG = config['TRAIN_IMG']
    TRAIN_ANNOT = config['TRAIN_ANNOT']
    
    # Collect all bounding box information
    all_widths = []
    all_heights = []
    all_areas = []
    all_aspect_ratios = []
    
    annotation_files = [f for f in os.listdir(TRAIN_ANNOT) if f.endswith('.xml')]
    
    # Limit samples for faster analysis
    if max_samples:
        annotation_files = annotation_files[:max_samples]
    
    print(f"Analyzing {len(annotation_files)} annotation files...")
    
    for ann_file in annotation_files:
        ann_path = os.path.join(TRAIN_ANNOT, ann_file)
        
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            # Get image dimensions
            size_elem = root.find('size')
            if size_elem is not None:
                img_width = int(size_elem.find('width').text)
                img_height = int(size_elem.find('height').text)
            else:
                # Fallback: try to get from image
                img_name = root.find('filename').text
                img_path = os.path.join(TRAIN_IMG, img_name)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    img_height, img_width = img.shape[:2]
                else:
                    print(f"Warning: Could not get dimensions for {ann_file}")
                    continue
            
            # Process all objects in this image
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                if bbox is not None:
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    # Calculate bbox properties in original image coordinates
                    width = xmax - xmin
                    height = ymax - ymin
                    area = width * height
                    aspect_ratio = width / height if height > 0 else 1.0
                    
                    # Normalize to [0, 1] range based on image size
                    norm_width = width / img_width
                    norm_height = height / img_height
                    norm_area = area / (img_width * img_height)
                    
                    all_widths.append(norm_width)
                    all_heights.append(norm_height)
                    all_areas.append(norm_area)
                    all_aspect_ratios.append(aspect_ratio)
                    
        except Exception as e:
            print(f"Error processing {ann_file}: {e}")
            continue
    
    print(f"Analyzed {len(all_widths)} bounding boxes")
    
    return {
        'widths': np.array(all_widths),
        'heights': np.array(all_heights),
        'areas': np.array(all_areas),
        'aspect_ratios': np.array(all_aspect_ratios)
    }


def generate_optimal_anchors(bbox_stats, target_resolution=[640, 640], num_feature_maps=6):
    """
    Generate optimal anchor configurations based on dataset analysis
    """
    widths = bbox_stats['widths']
    heights = bbox_stats['heights']
    areas = bbox_stats['areas']
    aspect_ratios = bbox_stats['aspect_ratios']
    
    print("\n" + "="*50)
    print("DATASET BBOX ANALYSIS")
    print("="*50)
    
    print(f"Normalized width range: {widths.min():.3f} - {widths.max():.3f}")
    print(f"Normalized height range: {heights.min():.3f} - {heights.max():.3f}")
    print(f"Normalized area range: {areas.min():.4f} - {areas.max():.3f}")
    print(f"Aspect ratio range: {aspect_ratios.min():.2f} - {aspect_ratios.max():.2f}")
    
    # Analyze aspect ratios using clustering
    aspect_ratios_clean = aspect_ratios[(aspect_ratios > 0.1) & (aspect_ratios < 10)]  # Remove outliers
    
    # Cluster aspect ratios to find common patterns
    ar_log = np.log(aspect_ratios_clean).reshape(-1, 1)
    kmeans_ar = KMeans(n_clusters=5, random_state=42)
    ar_clusters = kmeans_ar.fit_predict(ar_log)
    
    # Get representative aspect ratios
    unique_ars = []
    for i in range(5):
        cluster_ars = aspect_ratios_clean[ar_clusters == i]
        if len(cluster_ars) > 0:
            representative_ar = np.exp(kmeans_ar.cluster_centers_[i][0])
            unique_ars.append(representative_ar)
    
    unique_ars = sorted(unique_ars)
    print(f"Clustered aspect ratios: {[f'{ar:.2f}' for ar in unique_ars]}")
    
    # Analyze object sizes and map to feature maps
    # Convert normalized areas to actual pixel areas at target resolution
    pixel_areas = areas * (target_resolution[0] * target_resolution[1])
    pixel_sizes = np.sqrt(pixel_areas)  # Approximate object size (assuming square)
    
    # Define feature map scales (typical SSD scales)
    feature_map_sizes = [80, 40, 20, 10, 5, 3]  # For 640x640 input
    feature_map_scales = [target_resolution[0] / fm_size for fm_size in feature_map_sizes]
    
    print(f"\nFeature map scales: {[f'{scale:.1f}' for scale in feature_map_scales]}")
    
    # Map object sizes to appropriate feature maps
    size_percentiles = [10, 25, 40, 60, 80, 95]  # Percentiles for feature map assignment
    size_thresholds = np.percentile(pixel_sizes, size_percentiles)
    
    print(f"Object size thresholds: {[f'{t:.1f}' for t in size_thresholds]}")
    
    # Generate aspect ratios for each feature map
    optimal_aspect_ratios = []
    
    for fm_idx in range(num_feature_maps):
        # For smaller feature maps (detecting larger objects), use fewer aspect ratios
        # For larger feature maps (detecting smaller objects), use more aspect ratios
        
        if fm_idx in [0, 1]:  # First two feature maps - detect small objects
            # Use more aspect ratios for small object detection
            fm_aspect_ratios = [ar for ar in unique_ars if 0.3 <= ar <= 4.0][:4]
        elif fm_idx in [2, 3]:  # Middle feature maps - detect medium objects
            # Use moderate number of aspect ratios
            fm_aspect_ratios = [ar for ar in unique_ars if 0.4 <= ar <= 3.0][:3]
        else:  # Last feature maps - detect large objects
            # Use fewer aspect ratios for large object detection
            fm_aspect_ratios = [ar for ar in unique_ars if 0.5 <= ar <= 2.5][:2]
        
        # Ensure we have at least some aspect ratios
        if len(fm_aspect_ratios) == 0:
            fm_aspect_ratios = [1.0, 2.0]  # Default fallback
        
        # Round to reasonable precision and ensure 1.0 is included
        fm_aspect_ratios = list(set([round(ar, 1) for ar in fm_aspect_ratios] + [1.0]))
        fm_aspect_ratios.sort()
        
        optimal_aspect_ratios.append(fm_aspect_ratios)
    
    print(f"\nOptimal aspect ratios per feature map:")
    for i, ars in enumerate(optimal_aspect_ratios):
        print(f"  FM {i} ({feature_map_sizes[i]}x{feature_map_sizes[i]}): {ars}")
    
    return optimal_aspect_ratios, {
        'size_thresholds': size_thresholds,
        'feature_map_scales': feature_map_scales,
        'dataset_stats': {
            'mean_area': areas.mean(),
            'mean_aspect_ratio': aspect_ratios.mean(),
            'area_std': areas.std(),
            'aspect_ratio_std': aspect_ratios.std()
        }
    }


def visualize_bbox_analysis(bbox_stats):
    """
    Visualize the bounding box analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Area distribution
    axes[0, 0].hist(bbox_stats['areas'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Normalized Area')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Object Area Distribution')
    axes[0, 0].set_yscale('log')
    
    # Aspect ratio distribution
    ar_clean = bbox_stats['aspect_ratios'][(bbox_stats['aspect_ratios'] > 0.1) & 
                                          (bbox_stats['aspect_ratios'] < 10)]
    axes[0, 1].hist(ar_clean, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Aspect Ratio (Width/Height)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Aspect Ratio Distribution')
    axes[0, 1].set_yscale('log')
    
    # Width vs Height scatter
    axes[1, 0].scatter(bbox_stats['widths'], bbox_stats['heights'], alpha=0.5, s=1)
    axes[1, 0].set_xlabel('Normalized Width')
    axes[1, 0].set_ylabel('Normalized Height')
    axes[1, 0].set_title('Object Width vs Height')
    
    # Area vs Aspect Ratio
    axes[1, 1].scatter(bbox_stats['areas'], bbox_stats['aspect_ratios'], alpha=0.5, s=1)
    axes[1, 1].set_xlabel('Normalized Area')
    axes[1, 1].set_ylabel('Aspect Ratio')
    axes[1, 1].set_title('Area vs Aspect Ratio')
    axes[1, 1].set_ylim(0, 5)  # Limit y-axis for better visualization
    
    plt.tight_layout()
    plt.show()


def create_dynamic_anchor_generator(
    config_path, 
    target_resolution=[640, 640], 
    visualize=True,
    max_samples=1000
):
    """
    Main function to create optimal anchor generator for your dataset
    """
    print("Analyzing dataset for optimal anchor configuration...")
    
    # Analyze dataset
    bbox_stats = analyze_dataset_bboxes(config_path, max_samples=max_samples)
    
    if visualize:
        visualize_bbox_analysis(bbox_stats)
    
    # Generate optimal anchors
    optimal_aspect_ratios, analysis_info = generate_optimal_anchors(
        bbox_stats, target_resolution
    )
    
    return optimal_aspect_ratios, analysis_info


# Usage example:
if __name__ == "__main__":
    # Analyze your dataset and get optimal anchors
    config_path = args.config
    optimal_anchors, analysis = create_dynamic_anchor_generator(
        config_path, 
        target_resolution=[640, 640],
        visualize=True,
        max_samples=args.max_samples
    )
    
    print("\nOptimal anchor configuration:")
    for i, anchors in enumerate(optimal_anchors):
        print(f"Feature Map {i}: {anchors}")