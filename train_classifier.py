import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np

from tqdm.auto import tqdm
from src.img_cls.model import Dinov3Classification
from src.img_cls.datasets import get_datasets, get_data_loaders
from src.img_cls.utils import save_model, save_plots, SaveBestModel
from src.utils.common import get_dinov3_paths
from torch.optim.lr_scheduler import MultiStepLR

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', 
    type=int, 
    default=10,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-lr', '--learning-rate', 
    type=float,
    dest='learning_rate', 
    default=0.001,
    help='Learning rate for training the model'
)
parser.add_argument(
    '-b', '--batch-size',
    dest='batch_size',
    default=32,
    type=int
)
parser.add_argument(
    '--save-name',
    dest='save_name',
    default='model',
    help='file name of the final model to save'
)
parser.add_argument(
    '--fine-tune',
    dest='fine_tune',
    action='store_true',
    help='whether to fine-tune the model or train the classifier layer only'
)
parser.add_argument(
    '--out-dir',
    dest='out_dir',
    default='img_cls',
    help='output sub-directory path inside the `outputs` directory'
)
parser.add_argument(
    '--scheduler',
    type=int,
    nargs='+',
    default=[1000],
    help='number of epochs after which learning rate scheduler is applied'
)
parser.add_argument(
    '--train-dir',
    dest='train_dir',
    required=True,
    help='path to the training directory containing class folders in \
          PyTorch ImageFolder format'
)
parser.add_argument(
    '--valid-dir',
    dest='valid_dir',
    required=True,
    help='path to the validation directory containing class folders in \
          PyTorch ImageFolder format'
)
parser.add_argument(
    '--weights',
    help='path to the pretrained backbone weights',
    required=True
)
parser.add_argument(
    '--repo-dir',
    dest='repo_dir',
    help='path to the cloned DINOv3 repository',
)
parser.add_argument(
    '--model-name',
    dest='model_name',
    help='name of the model, check: https://github.com/facebookresearch/dinov3?tab=readme-ov-file#pretrained-backbones-via-pytorch-hub',
    default='dinov3_vits16'
)
args = parser.parse_args()
print(args)

DINOV3_REPO, DINOV3_WEIGHTS = get_dinov3_paths()

# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# Validation function.
def validate(model, testloader, criterion, class_names):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    # Create a directory with the model name for outputs.
    out_dir = os.path.join('outputs', args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets(
        train_dir=args.train_dir, 
        valid_dir=args.valid_dir
    )
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Classes: {dataset_classes}")
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(
        dataset_train, dataset_valid, batch_size=args.batch_size
    )

    # Learning_parameters. 
    lr = args.learning_rate
    epochs = args.epochs
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    # Load the model.
    model = Dinov3Classification(
        num_classes=len(dataset_classes), 
        fine_tune=args.fine_tune, 
        weights=os.path.join(DINOV3_WEIGHTS, args.weights),
        model_name=args.model_name,
        repo_dir=DINOV3_REPO
    ).to(device)
    print(model)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, nesterov=True
    )
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()

    # Scheduler.
    scheduler = MultiStepLR(optimizer, milestones=args.scheduler, gamma=0.1)

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                    criterion, dataset_classes)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        save_best_model(
            valid_epoch_loss, epoch, model, out_dir, args.save_name
        )
        print('-'*50)
        scheduler.step()
        last_lr = scheduler.get_last_lr()
        print(f"LR for next epoch: {last_lr}")

    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion, out_dir, args.save_name)
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, out_dir)
    print('TRAINING COMPLETE')