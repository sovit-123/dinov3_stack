import torch
import cv2
import numpy as np
import os
import glob as glob

from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader
from src.detection.custom_utils import collate_fn, get_train_transform, get_valid_transform

# The dataset class.
class CustomDataset(Dataset):
    def __init__(
            self, 
            img_path, 
            annot_path, 
            width, 
            height, 
            classes, 
            transforms=None
    ):
        self.transforms = transforms
        self.img_path = img_path
        self.annot_path = annot_path
        self.height = height
        self.width = width
        self.classes = classes
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []
        
        # Get all the image paths in sorted order.
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.img_path, file_type)))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # Capture the image name and the full image path.
        image_name = self.all_images[idx]
        image_path = os.path.join(self.img_path, image_name)

        # Read and preprocess the image.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        # Capture the corresponding XML file for getting the annotations.
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.annot_path, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # Original image width and height.
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        # Box coordinates for xml files are extracted 
        # and corrected for image size given.
        for member in root.findall('object'):
            # Get label and map the `classes`.
            labels.append(self.classes.index(member.find('name').text))
            
            # Left corner x-coordinates.
            xmin = int(member.find('bndbox').find('xmin').text)
            # Right corner x-coordinates.
            xmax = int(member.find('bndbox').find('xmax').text)
            # Left corner y-coordinates.
            ymin = int(member.find('bndbox').find('ymin').text)
            # Right corner y-coordinates.
            ymax = int(member.find('bndbox').find('ymax').text)
            
            # Resize the bounding boxes according 
            # to resized image `width`, `height`.
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height

            # Check that max coordinates are at least one pixel
            # larger than min coordinates.
            if xmax_final == xmin_final:
                xmin_final -= 1
            if ymax_final == ymin_final:
                ymin_final -= 1
            # Check that all coordinates are within the image.
            if xmax_final > self.width:
                xmax_final = self.width
            if ymax_final > self.height:
                ymax_final = self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # Bounding box to tensor.
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Area of the bounding boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
            else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # Labels to tensor.

        # Prepare the final `target` dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # Apply the image transforms.
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            target['labels'] = torch.as_tensor(sample['labels'], dtype=torch.int64)
        
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

# Prepare the final datasets and data loaders.
def create_train_dataset(img_dir, annot_dir, classes, resize=(640, 640)):
    train_dataset = CustomDataset(
        img_dir, 
        annot_dir, 
        resize[0], 
        resize[1], 
        classes, 
        get_train_transform()
    )
    return train_dataset
def create_valid_dataset(img_dir, annot_dir, classes, resize=(640, 640)):
    valid_dataset = CustomDataset(
        img_dir, 
        annot_dir, 
        resize[0], 
        resize[1], 
        classes, 
        get_valid_transform()
    )
    return valid_dataset
def create_train_loader(train_dataset, batch_size, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True
    )
    return train_loader
def create_valid_loader(valid_dataset, batch_size, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True
    )
    return valid_loader


# execute `datasets.py` using Python command from 
# Terminal to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    import yaml
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        help='path to the configuration yaml file in detection_configs folder',
        default='detection_configs/voc.yaml'
    )
    args = parser.parse_args()
    print(args)

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    TRAIN_IMG =  config['TRAIN_IMG']
    TRAIN_ANNOT = config['TRAIN_ANNOT']
    VALID_IMG = config['VALID_IMG']
    VALID_ANNOT = config['VALID_ANNOT']
    CLASSES = config['CLASSES']

    # sanity check of the Dataset pipeline with sample visualization
    dataset = CustomDataset(
        TRAIN_IMG, 
        TRAIN_ANNOT, 
        640, 
        640, 
        CLASSES,
        get_train_transform()
    )
    print(f"Number of training images: {len(dataset)}")
    
    # function to visualize a single sample
    def visualize_sample(image, target):
        image = np.array(image).transpose((1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 0, 255), 
                2
            )
            cv2.putText(
                image, 
                label, 
                (int(box[0]), int(box[1]-5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        
    NUM_SAMPLES_TO_VISUALIZE = 50
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)