import glob
import albumentations as A
import cv2
import torch
import numpy as np

from src.img_seg.utils import get_label_mask, set_class_values
from torch.utils.data import Dataset, DataLoader

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)

def get_images(train_images, train_masks, valid_images, valid_masks):
    train_images = glob.glob(f"{train_images}/*")
    train_images.sort()
    train_masks = glob.glob(f"{train_masks}/*")
    train_masks.sort()
    valid_images = glob.glob(f"{valid_images}/*")
    valid_images.sort()
    valid_masks = glob.glob(f"{valid_masks}/*")
    valid_masks.sort()

    return train_images, train_masks, valid_images, valid_masks

# TODO: Batchwise rescaling with different image ratios.
def train_transforms(img_size):
    """
    Transforms/augmentations for training images and masks.

    :param img_size: Integer, for image resize.
    """
    train_image_transform = A.Compose([
        A.Resize(
            img_size[1], 
            img_size[0], 
            always_apply=True,
            # interpolation=cv2.INTER_CUBIC
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=25),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD, max_pixel_value=255.)
    ], is_check_shapes=False)
    return train_image_transform

# TODO: Batchwise rescaling with different image ratios.
def valid_transforms(img_size):
    """
    Transforms/augmentations for validation images and masks.

    :param img_size: Integer, for image resize.
    """
    valid_image_transform = A.Compose([
        A.Resize(
            img_size[1], img_size[0], 
            always_apply=True, 
            # interpolation=cv2.INTER_CUBIC
        ),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD, max_pixel_value=255.)
    ], is_check_shapes=False)
    return valid_image_transform

class SegmentationDataset(Dataset):
    def __init__(
        self, 
        image_paths, 
        mask_paths, 
        tfms, 
        label_colors_list,
        classes_to_train,
        all_classes
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        self.class_values = set_class_values(
            self.all_classes, self.classes_to_train
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype('float32')

        # Make all pixel > 0 as 255.
        if len(self.all_classes) == 2:
            im = mask > 0
            mask[im] = 255
            mask[np.logical_not(im)] = 0

        transformed = self.tfms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Get 2D label mask.
        mask = get_label_mask(mask, self.class_values, self.label_colors_list).astype('uint8')
        # mask = Image.fromarray(mask)

        # To C, H, W.
        image = image.transpose(2, 0, 1)

        return torch.tensor(image), torch.LongTensor(mask)
    
def collate_fn(inputs):
    batch = dict()
    batch[0] = torch.stack([i[0] for i in inputs], dim=0)
    batch[1] = torch.stack([i[1] for i in inputs], dim=0)

    return batch

def get_dataset(
    train_image_paths, 
    train_mask_paths,
    valid_image_paths,
    valid_mask_paths,
    all_classes,
    classes_to_train,
    label_colors_list,
    img_size
):
    train_tfms = train_transforms(img_size)
    valid_tfms = valid_transforms(img_size)

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    valid_dataset = SegmentationDataset(
        valid_image_paths,
        valid_mask_paths,
        valid_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    return train_dataset, valid_dataset

def get_data_loaders(train_dataset, valid_dataset, batch_size):
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        drop_last=False, 
        num_workers=8,
        shuffle=True,
        collate_fn=collate_fn
    )
    valid_data_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        drop_last=False, 
        num_workers=8,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_data_loader, valid_data_loader