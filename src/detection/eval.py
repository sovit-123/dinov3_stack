import torch

from tqdm import tqdm
from src.detection.config import (
    DEVICE, 
    NUM_CLASSES, 
    NUM_WORKERS, 
    RESIZE_TO,
    VALID_ANNOT,
    VALID_IMG,
    CLASSES,
    BATCH_SIZE
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.detection.model import faster_vit_0_any_res
from src.detection.datasets import create_valid_dataset, create_valid_loader

# Evaluation function
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
    # Load the best model and trained weights.
    model = faster_vit_0_any_res(
        pretrained=False, 
        num_classes=NUM_CLASSES, 
        resolution=(RESIZE_TO, RESIZE_TO)
    )
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    test_dataset = create_valid_dataset(
        VALID_IMG, VALID_ANNOT, CLASSES, RESIZE_TO
    )
    test_loader = create_valid_loader(
        valid_dataset=test_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )

    metric_summary = validate(test_loader, model)
    print(f"mAP_50: {metric_summary['map_50']*100:.3f}")
    print(f"mAP_50_95: {metric_summary['map']*100:.3f}")