import torch
import torch.nn as nn

from contextlib import nullcontext
from tqdm import tqdm
from typing import List, Optional

from src.img_seg.utils import draw_translucent_seg_maps
from src.img_seg.metrics import IOUEval

def _run_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int,
    is_train: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    save_dir: Optional[str] = None,
    label_colors_list: Optional[List] = None,
    viz_map: Optional[List] = None,
):
    """
    Runs one epoch of training or validation.

    :param model: The neural network model.
    :param dataloader: The DataLoader for the current epoch.
    :param device: The computation device.
    :param criterion: The loss function.
    :param num_classes: Number of classes for segmentation.
    :param is_train: Boolean, True if it's a training epoch, False for validation.
    :param optimizer: The optimizer for training. Required if is_train is True.
    :param epoch: The current epoch number. Required for validation visualization.
    :param save_dir: Directory to save validation visualizations.
    :param label_colors_list: List of colors for labels.
    :param viz_map: Visualization map for segmentation.
    """
    model.train() if is_train else model.eval()

    running_loss = 0.0
    iou_eval = IOUEval(num_classes)
    prog_bar = tqdm(
        dataloader,
        total=len(dataloader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )

    context = nullcontext() if is_train else torch.no_grad()

    with context:
        for i, data in enumerate(prog_bar):
            pixel_values, target = data[0].to(device), data[1].to(device)

            if is_train:
                assert optimizer is not None, "Optimizer must be provided for training"
                optimizer.zero_grad()

            outputs = model(pixel_values)

            upsampled_logits = nn.functional.interpolate(
                outputs, size=target.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            loss = criterion(upsampled_logits, target)
            running_loss += loss.item()

            if is_train:
                loss.backward()
                optimizer.step()

            iou_eval.addBatch(upsampled_logits.max(1)[1].data, target.data)

            # For validation, save the first batch's segmentation map.
            if not is_train and i == 0 and all(p is not None for p in [epoch, save_dir, label_colors_list, viz_map]):
                draw_translucent_seg_maps(
                    pixel_values,
                    upsampled_logits,
                    epoch,
                    i,
                    save_dir,
                    label_colors_list,
                    viz_map
                )

    epoch_loss = running_loss / len(dataloader)
    overall_acc, _, _, mIOU = iou_eval.getMetric()
    return epoch_loss, overall_acc, mIOU

def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    classes_to_train: List[str]
):
    print('Training')
    return _run_one_epoch(
        model=model,
        dataloader=train_dataloader,
        device=device,
        criterion=criterion,
        num_classes=len(classes_to_train),
        is_train=True,
        optimizer=optimizer
    )

def validate(
    model: nn.Module,
    valid_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    classes_to_train: List[str],
    label_colors_list: List,
    epoch: int,
    save_dir: str,
    viz_map: List
):
    print('Validating')
    return _run_one_epoch(
        model=model,
        dataloader=valid_dataloader,
        device=device,
        criterion=criterion,
        num_classes=len(classes_to_train),
        is_train=False,
        epoch=epoch,
        save_dir=save_dir,
        label_colors_list=label_colors_list,
        viz_map=viz_map
    )