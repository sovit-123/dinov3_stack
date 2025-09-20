"""
Script for image classification inference using trained model.

USAGE:
python infer_classifier.py --weights <path to the weights.pth file> \
--input <directory containing inference images>

Update the `CLASS_NAMES` list to contain the trained class names. 
"""

import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
import glob
import argparse
import pathlib
import yaml

from src.img_cls.model import Dinov3Classification
from src.utils.common import get_dinov3_paths

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-w', '--weights', 
    default='../outputs/best_model.pth',
    help='path to the model weights',
)
parser.add_argument(
    '--input',
    required=True,
    help='directory containing images for inference'
)
parser.add_argument(
    '--config',
    required=True,
    help='path to the config file containing the class names'
)
parser.add_argument(
    '--model-name',
    dest='model_name',
    help='name of the model, check: https://github.com/facebookresearch/dinov3?tab=readme-ov-file#pretrained-backbones-via-pytorch-hub',
    default='dinov3_vits16'
)
parser.add_argument(
    '--repo-dir',
    dest='repo_dir',
    help='path to the cloned DINOv3 repository'
)
parser.add_argument(
    '--out-dir',
    dest='out_dir',
    default='inference_results_classifier',
    help='output sub-directory path inside the `outputs` directory'
)
args = parser.parse_args()

DINOV3_REPO, DINOV3_WEIGHTS = get_dinov3_paths()

# Constants and other configurations.
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 256

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    f.close()
CLASS_NAMES = config['CLASS_NAMES']

# Validation transforms
def get_test_transform(image_size):
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return test_transform

def annotate_image(output_class, orig_image):
    class_name = CLASS_NAMES[int(output_class)]
    cv2.putText(
        orig_image, 
        f"{class_name}", 
        (5, 35), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (0, 0, 255), 
        2, 
        lineType=cv2.LINE_AA
    )
    return orig_image

def inference(model, testloader, device, orig_image):
    """
    Function to run inference.

    :param model: The trained model.
    :param testloader: The test data loader.
    :param DEVICE: The computation device.
    """
    model.eval()
    counter = 0

    with torch.no_grad():
        counter += 1
        image = testloader
        image = image.to(device)

        # Forward pass.
        outputs = model(image)

    # Softmax probabilities.
    predictions = F.softmax(outputs, dim=1).cpu().numpy()
    # Predicted class number.
    output_class = np.argmax(predictions)
    # Show and save the results.
    result = annotate_image(output_class, orig_image)
    return result

if __name__ == '__main__':
    weights_path = pathlib.Path(args.weights)
    infer_result_path = os.path.join(
        'outputs', args.out_dir
    )
    os.makedirs(infer_result_path, exist_ok=True)

    checkpoint = torch.load(weights_path)
    # Load the model.
    model = Dinov3Classification(
        num_classes=len(CLASS_NAMES), 
        model_name=args.model_name,
        repo_dir=DINOV3_REPO
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    all_image_paths = glob.glob(os.path.join(args.input, '*'))

    transform = get_test_transform(IMAGE_RESIZE)

    for i, image_path in enumerate(all_image_paths):
        print(f"Inference on image: {i+1}")
        image = cv2.imread(image_path)
        orig_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        result = inference(
            model, 
            image,
            DEVICE,
            orig_image
        )
        # Save the image to disk.
        image_name = image_path.split(os.path.sep)[-1]
        # cv2.imshow('Image', result)
        # cv2.waitKey(1)
        cv2.imwrite(
            os.path.join(infer_result_path, image_name), result
        )