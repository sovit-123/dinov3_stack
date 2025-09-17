# from src.detection.model import Dinov3Detection
from src.detection.model import dinov3_detection
from src.detection.custom_utils import (
    Averager, 
    SaveBestModel, 
    save_model, 
    save_loss_plot,
    save_mAP
)
from tqdm.auto import tqdm
from src.detection.datasets import (
    create_train_dataset, 
    create_valid_dataset, 
    create_train_loader, 
    create_valid_loader
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import MultiStepLR
from src.utils.common import get_dinov3_paths

import torch
import matplotlib.pyplot as plt
import time
import os
import argparse
import random
import numpy as np
import torch.multiprocessing as mp
import yaml

# Set sharing strategy
torch.multiprocessing.set_sharing_strategy('file_system')

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
    default=[640, 640],
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
    default=[1000],
    nargs='+',
    type=int
)
parser.add_argument(
    '--out-dir',
    dest='out_dir',
    default='img_det',
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
parser.add_argument(
    '--workers',
    default=4,
    type=int,
    help='number of parllel workers for the data loader'
)
parser.add_argument(
    '--optimizer',
    default='AdamW',
    choices=['SGD', 'AdamW']
)
parser.add_argument(
    '--config',
    help='path to the configuration yaml file in detection_configs folder',
    default='detection_configs/voc.yaml'
)
parser.add_argument(
    '--head',
    default='retinanet',
    choices=['ssd', 'retinanet'],
    help='whether to build with SSD or RetinaNet detection head'
)
args = parser.parse_args()
print(args)

# torch.multiprocessing.set_sharing_strategy('file_system')

plt.style.use('ggplot')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

TRAIN_IMG =  config['TRAIN_IMG']
TRAIN_ANNOT = config['TRAIN_ANNOT']
VALID_IMG = config['VALID_IMG']
VALID_ANNOT = config['VALID_ANNOT']
CLASSES = config['CLASSES']
NUM_CLASSES = len(CLASSES)
VISUALIZE_TRANSFORMED_IMAGES = config['VISUALIZE_TRANSFORMED_IMAGES']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DINOV3_REPO, DINOV3_WEIGHTS = get_dinov3_paths()

# Function for running training iterations.
def train(train_data_loader, model):
    print('Training')
    model.train()
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return loss_value

# Function for running validation iterations.
def validate(valid_data_loader, model):
    print('Validating')
    model.eval()
    
    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images, targets)

        # For mAP calculation using Torchmetrics.
        #####################################
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
        #####################################

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary

if __name__ == '__main__':
    OUT_DIR = os.path.join('outputs', args.out_dir)
    os.makedirs(OUT_DIR, exist_ok=True)
    train_dataset = create_train_dataset(
        TRAIN_IMG, TRAIN_ANNOT, CLASSES, args.imgsz,
    )
    valid_dataset = create_valid_dataset(
        VALID_IMG, VALID_ANNOT, CLASSES, args.imgsz
    )
    train_loader = create_train_loader(
        train_dataset=train_dataset, 
        batch_size=args.batch, 
        num_workers=args.workers
    )
    valid_loader = create_valid_loader(
        valid_dataset=valid_dataset, 
        batch_size=args.batch, 
        num_workers=args.workers
    )
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # model = Dinov3Detection(
    #     fine_tune=args.fine_tune,
    #     num_classes=NUM_CLASSES, 
    #     weights=os.path.join(DINOV3_WEIGHTS, args.weights),
    #     model_name=args.model_name,
    #     repo_dir=DINOV3_REPO,
    #     feature_extractor=args.feature_extractor
    # )

    model = dinov3_detection(
        fine_tune=args.fine_tune,
        num_classes=NUM_CLASSES, 
        weights=os.path.join(DINOV3_WEIGHTS, args.weights),
        model_name=args.model_name,
        repo_dir=DINOV3_REPO,
        feature_extractor=args.feature_extractor,
        head=args.head
    )

    model = model.to(DEVICE)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    params = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=0.9, nesterov=True
        )
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=args.lr)

    # optimizer = getattr(torch.optim, args.optimizer)(params, lr=args.lr, momentum=0.9, nesterov=True)
    print(f"Using {optimizer} for {args.model_name}")

    scheduler = MultiStepLR(
        optimizer=optimizer, 
        milestones=args.scheduler_epochs, 
        gamma=0.1, 
        # verbose=True
    )

    # To monitor training loss
    train_loss_hist = Averager()
    # To store training loss and mAP values.
    train_loss_list = []
    map_50_list = []
    map_list = []

    # Mame to save the trained model with.
    MODEL_NAME = 'model'

    # Whether to show transformed images from data loader or not.
    if VISUALIZE_TRANSFORMED_IMAGES:
        from src.detection.custom_utils import show_tranformed_image
        show_tranformed_image(train_loader, device=DEVICE, classes=CLASSES)

    # To save best model.
    save_best_model = SaveBestModel()

    # Training loop.
    for epoch in range(args.epochs):
        print(f"\nEPOCH {epoch+1} of {args.epochs}")

        # Reset the training loss histories for the current epoch.
        train_loss_hist.reset()

        # Start timer and carry out training and validation.
        start = time.time()
        train_loss = train(train_loader, model)
        metric_summary = validate(valid_loader, model)
        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch+1} mAP@0.50:0.95: {metric_summary['map']}")
        print(f"Epoch #{epoch+1} mAP@0.50: {metric_summary['map_50']}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        train_loss_list.append(train_loss)
        map_50_list.append(metric_summary['map_50'])
        map_list.append(metric_summary['map'])

        # save the best model till now.
        save_best_model(
            model, float(metric_summary['map']), epoch, OUT_DIR
        )
        # Save the current epoch model.
        save_model(epoch, model, optimizer, out_dir=OUT_DIR)

        # Save loss plot.
        save_loss_plot(OUT_DIR, train_loss_list)

        # Save mAP plot.
        save_mAP(OUT_DIR, map_50_list, map_list)
        scheduler.step()
        last_lr = scheduler.get_last_lr()
        print(f"LR for next epoch: {last_lr}")