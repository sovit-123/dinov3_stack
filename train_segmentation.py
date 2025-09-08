import torch
import os
import argparse
import torch.nn as nn
import yaml
import numpy as np
import random

from src.img_seg.datasets import get_images, get_dataset, get_data_loaders
from src.img_seg.model import Dinov3Segmentation
from src.img_seg.engine import train, validate
from src.img_seg.utils import save_model, SaveBestModel, save_plots, SaveBestModelIOU
from src.utils.common import get_dinov3_paths
from torch.optim.lr_scheduler import MultiStepLR
from torchinfo import summary

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument(
    '--epochs',
    default=10,
    help='number of epochs to train for',
    type=int
)
parser.add_argument(
    '--lr',
    default=0.0001,
    help='learning rate for optimizer',
    type=float
)
parser.add_argument(
    '--batch',
    default=4,
    help='batch size for data loader',
    type=int
)
parser.add_argument(
    '--imgsz', 
    default=[448, 448],
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--scheduler',
    action='store_true',
)
parser.add_argument(
    '--scheduler-epochs',
    dest='scheduler_epochs',
    default=[30],
    nargs='+',
    type=int
)
parser.add_argument(
    '--train-images',
    dest='train_images',
    required=True,
    help='path to training images'
)
parser.add_argument(
    '--train-masks',
    dest='train_masks',
    required=True,
    help='path to training masks'
)
parser.add_argument(
    '--valid-images',
    dest='valid_images',
    required=True,
    help='path to validation images'
)
parser.add_argument(
    '--valid-masks',
    dest='valid_masks',
    required=True,
    help='path to validation masks'
)
parser.add_argument(
    '--config',
    required=True,
    help='path to the dataset configuration file'
)
parser.add_argument(
    '--out-dir',
    dest='out_dir',
    default='img_seg',
    help='output sub-directory path inside the `outputs` directory'
)
parser.add_argument(
    '--weights',
    help='path to the pretrained backbone weights',
    required=True
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
    '--fine-tune',
    dest='fine_tune',
    action='store_true'
)
parser.add_argument(
    '--feautre-extractor',
    dest='feature_extractor',
    default='multi',
    choices=['last', 'multi'],
    help='whether to use layer or multiple layers as features'
)
args = parser.parse_args()
print(args)

DINOV3_REPO, DINOV3_WEIGHTS = get_dinov3_paths()

if __name__ == '__main__':
    # Create a directory with the model name for outputs.
    out_dir = os.path.join('outputs', args.out_dir)
    out_dir_valid_preds = os.path.join(out_dir, 'valid_preds')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_valid_preds, exist_ok=True)

    # Set configurations.
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    ALL_CLASSES = config['ALL_CLASSES']
    LABEL_COLORS_LIST = config['LABEL_COLORS_LIST']
    VIZ_MAP = config['VIS_LABEL_MAP']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Dinov3Segmentation(
        fine_tune=args.fine_tune,
        num_classes=len(ALL_CLASSES), 
        weights=os.path.join(DINOV3_WEIGHTS, args.weights),
        model_name=args.model_name,
        repo_dir=DINOV3_REPO,
        feature_extractor=args.feature_extractor
    )
    _ = model.to(device)
    summary(
        model, 
        (1, 3, 448, 448),
        col_names=('input_size', 'output_size', 'num_params'),
        row_settings=['var_names']
    )

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.0001, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_images, train_masks, valid_images, valid_masks = get_images(
        train_images=args.train_images,
        train_masks=args.train_masks,
        valid_images=args.valid_images,
        valid_masks=args.valid_masks,  
    )

    train_dataset, valid_dataset = get_dataset(
        train_images, 
        train_masks,
        valid_images,
        valid_masks,
        ALL_CLASSES,
        ALL_CLASSES,
        LABEL_COLORS_LIST,
        img_size=args.imgsz
    )

    train_dataloader, valid_dataloader = get_data_loaders(
        train_dataset, 
        valid_dataset,
        args.batch
    )

    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()
    save_best_iou = SaveBestModelIOU()
    # LR Scheduler.
    scheduler = MultiStepLR(
        optimizer, milestones=args.scheduler_epochs, gamma=0.1
    )

    train_loss, train_pix_acc, train_miou = [], [], []
    valid_loss, valid_pix_acc, valid_miou = [], [], []
    
    for epoch in range (args.epochs):
        print(f"EPOCH: {epoch + 1}")
        train_epoch_loss, train_epoch_pixacc, train_epoch_miou = train(
            model,
            train_dataloader,
            device,
            optimizer,
            criterion,
            ALL_CLASSES
        )
        valid_epoch_loss, valid_epoch_pixacc, valid_epoch_miou = validate(
            model,
            valid_dataloader,
            device,
            criterion,
            ALL_CLASSES,
            LABEL_COLORS_LIST,
            epoch,
            save_dir=out_dir_valid_preds,
            viz_map=VIZ_MAP
        )
        train_loss.append(train_epoch_loss)
        train_pix_acc.append(train_epoch_pixacc)
        train_miou.append(train_epoch_miou)
        valid_loss.append(valid_epoch_loss)
        valid_pix_acc.append(valid_epoch_pixacc)
        valid_miou.append(valid_epoch_miou)

        save_best_model(
            valid_epoch_loss, epoch, model, out_dir, name='model_loss'
        )
        save_best_iou(
            valid_epoch_miou, epoch, model, out_dir, name='model_iou'
        )

        print(
            f"Train Epoch Loss: {train_epoch_loss:.4f},",
            f"Train Epoch PixAcc: {train_epoch_pixacc:.4f},",
            f"Train Epoch mIOU: {train_epoch_miou:4f}"
        )
        print(
            f"Valid Epoch Loss: {valid_epoch_loss:.4f},", 
            f"Valid Epoch PixAcc: {valid_epoch_pixacc:.4f}",
            f"Valid Epoch mIOU: {valid_epoch_miou:4f}"
        )

        if args.scheduler:
            scheduler.step()
        last_lr = scheduler.get_last_lr()
        print(f"LR for next epoch: {last_lr}")
        
        print('-' * 50)

    # Save the loss and accuracy plots.
    save_plots(
        train_pix_acc, valid_pix_acc, 
        train_loss, valid_loss,
        train_miou, valid_miou, 
        out_dir
    )
    # Save final model.
    save_model(args.epochs, model, optimizer, criterion, out_dir, name='final_model')
    print('TRAINING COMPLETE')