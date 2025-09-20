"""
Script to get dataset statistics.

USAGE (execute from parent project directory):
python -m src.detection.analyze_dataset
"""

import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import yaml

from tqdm import tqdm
from collections import defaultdict, Counter

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
TRAIN_ANNOT, VALID_ANNOT = config['TRAIN_ANNOT'], config['VALID_ANNOT']

def parse_voc_annotation(annotation_path):
    """Parse a VOC XML annotation file and extract objects"""
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    # Extract image dimensions
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Extract objects
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        # Skip if class not in our classes list
        if name not in CLASSES:
            continue
            
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        # Calculate normalized coordinates and area
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        area = bbox_width * bbox_height
        norm_area = area / (width * height)
        
        objects.append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'width': bbox_width,
            'height': bbox_height,
            'area': area,
            'norm_area': norm_area,
            'aspect_ratio': bbox_width / bbox_height if bbox_height > 0 else 0
        })
    
    return {
        'filename': root.find('filename').text,
        'width': width,
        'height': height,
        'objects': objects,
        'num_objects': len(objects)
    }

def analyze_dataset(annotation_dir):
    """Analyze a directory of VOC annotations"""
    # Get all XML files
    xml_files = [os.path.join(annotation_dir, f) for f in os.listdir(annotation_dir) 
                 if f.endswith('.xml')]
    
    print(f"Found {len(xml_files)} annotation files")
    
    # Initialize counters
    class_counts = defaultdict(int)
    images_with_class = defaultdict(int)
    bbox_areas = []
    aspect_ratios = []
    objects_per_image = []
    image_sizes = []
    
    # Process each annotation file
    for xml_file in tqdm(xml_files, desc="Processing annotations"):
        try:
            annotation = parse_voc_annotation(xml_file)
            image_sizes.append((annotation['width'], annotation['height']))
            objects_per_image.append(annotation['num_objects'])
            
            # Track which classes appear in this image
            classes_in_image = set()
            
            for obj in annotation['objects']:
                class_name = obj['name']
                class_counts[class_name] += 1
                classes_in_image.add(class_name)
                bbox_areas.append(obj['norm_area'])
                aspect_ratios.append(obj['aspect_ratio'])
            
            # Update images containing each class
            for class_name in classes_in_image:
                images_with_class[class_name] += 1
                
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
    
    return {
        'class_counts': dict(class_counts),
        'images_with_class': dict(images_with_class),
        'bbox_areas': bbox_areas,
        'aspect_ratios': aspect_ratios,
        'objects_per_image': objects_per_image,
        'image_sizes': image_sizes,
        'total_images': len(xml_files)
    }

def print_statistics(stats, dataset_name):
    """Print statistics in a formatted way"""
    print("\n" + "="*60)
    print(f"{dataset_name.upper()} DATASET STATISTICS")
    print("="*60)
    
    print(f"Total images: {stats['total_images']:,}")
    print(f"Total objects: {sum(stats['class_counts'].values()):,}")
    print(f"Average objects per image: {np.mean(stats['objects_per_image']):.2f}")
    
    # Image size statistics
    widths, heights = zip(*stats['image_sizes'])
    print(f"Average image size: {np.mean(widths):.1f}×{np.mean(heights):.1f}")
    print(f"Min image size: {min(widths)}×{min(heights)}")
    print(f"Max image size: {max(widths)}×{max(heights)}")
    
    # Class distribution
    print("\n" + "-"*60)
    print("CLASS DISTRIBUTION")
    print("-"*60)
    
    # Sort classes by count
    sorted_classes = sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True)
    
    for i, (class_name, count) in enumerate(sorted_classes, 1):
        images_with_class = stats['images_with_class'][class_name]
        percentage = (count / sum(stats['class_counts'].values())) * 100
        print(f"{i:2d}. {class_name:15s}: {count:5d} objects ({percentage:5.2f}%) "
              f"in {images_with_class:4d} images")
    
    # Bounding box statistics
    print("\n" + "-"*60)
    print("BOUNDING BOX STATISTICS")
    print("-"*60)
    print(f"Average normalized area: {np.mean(stats['bbox_areas']):.4f}")
    print(f"Min normalized area: {np.min(stats['bbox_areas']):.6f}")
    print(f"Max normalized area: {np.max(stats['bbox_areas']):.4f}")
    print(f"Average aspect ratio: {np.mean(stats['aspect_ratios']):.2f}")
    
    # Objects per image statistics
    print("\n" + "-"*60)
    print("OBJECTS PER IMAGE")
    print("-"*60)
    print(f"Min objects: {np.min(stats['objects_per_image'])}")
    print(f"Max objects: {np.max(stats['objects_per_image'])}")
    print(f"Images with 0 objects: {stats['objects_per_image'].count(0)}")
    print(f"Images with 1 object: {stats['objects_per_image'].count(1)}")
    print(f"Images with 2-5 objects: {sum(1 for x in stats['objects_per_image'] if 2 <= x <= 5)}")
    print(f"Images with 6+ objects: {sum(1 for x in stats['objects_per_image'] if x >= 6)}")

def main():
    # Analyze training set
    train_stats = analyze_dataset(TRAIN_ANNOT)
    print_statistics(train_stats, "TRAINING")
    
    # Analyze validation set
    valid_stats = analyze_dataset(VALID_ANNOT)
    print_statistics(valid_stats, "VALIDATION")
    
    # Combined statistics
    combined_class_counts = defaultdict(int)
    for class_name in set(train_stats['class_counts'].keys()) | set(valid_stats['class_counts'].keys()):
        combined_class_counts[class_name] = train_stats['class_counts'].get(class_name, 0) + \
                                           valid_stats['class_counts'].get(class_name, 0)
    
    print("\n" + "="*60)
    print("COMBINED DATASET STATISTICS")
    print("="*60)
    print(f"Total images: {train_stats['total_images'] + valid_stats['total_images']:,}")
    print(f"Total objects: {sum(combined_class_counts.values()):,}")
    
    # Check for missing classes
    all_classes = set(CLASSES[1:])  # Exclude background
    detected_classes = set(combined_class_counts.keys())
    missing_classes = all_classes - detected_classes
    
    if missing_classes:
        print(f"\n⚠️  WARNING: Missing classes in dataset: {', '.join(missing_classes)}")
    else:
        print("\n✅ All classes are represented in the dataset")

if __name__ == "__main__":
    main()