import torch
import torch.nn as nn
import math

from collections import OrderedDict
from torchinfo import summary


def load_model(weights: str=None, model_name: str=None, repo_dir: str=None):
    if weights is not None:
        print('Loading pretrained backbone weights from: ', weights)
        model = torch.hub.load(
            repo_dir, 
            model_name, 
            source='local', 
            weights=weights
        )
    else:
        print('No pretrained weights path given. Loading with random weights.')
        model = torch.hub.load(
            repo_dir, 
            model_name, 
            source='local'
        )
    
    return model

class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, nc=1):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, nc, kernel_size=1)
        )

    def forward(self, x):
        return self.decode(x)

class Dinov3Segmentation(nn.Module):
    def __init__(
        self, 
        fine_tune: bool=False, 
        num_classes: int=2,
        weights: str=None,
        model_name: str=None,
        repo_dir: str=None
    ):
        super(Dinov3Segmentation, self).__init__()

        self.backbone_model = load_model(
            weights=weights, model_name=model_name, repo_dir=repo_dir
        )
        self.num_classes = num_classes

        if fine_tune:
            for name, param in self.backbone_model.named_parameters():
                param.requires_grad = True
        else:
            for name, param in self.backbone_model.named_parameters():
                param.requires_grad = False

        self.decode_head = SimpleDecoder(
            in_channels=self.backbone_model.norm.normalized_shape[0], 
            nc=self.num_classes
        )

        self.model = nn.Sequential(OrderedDict([
            ('backbone', self.backbone_model),
            ('decode_head', self.decode_head)
        ]))

    def forward(self, x):
        # Backbone forward pass
        features = self.model.backbone.get_intermediate_layers(
            x, 
            n=1, 
            reshape=True, 
            return_class_token=False, 
            norm=True
        )[0]

        # Decoder forward pass
        classifier_out = self.model.decode_head(features)
        return classifier_out
    
if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    from src.utils.common import get_dinov3_paths

    import numpy as np
    import os

    DINOV3_REPO, DINOV3_WEIGHTS = get_dinov3_paths()

    input_size = 640

    transform = transforms.Compose([
        transforms.Resize(
            input_size, 
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        )
    ])

    model = Dinov3Segmentation(
        repo_dir=DINOV3_REPO, 
        weights=os.path.join(DINOV3_WEIGHTS, 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth'),
        model_name='dinov3_vits16'
    )
    model.eval()
    print(model)

    random_image = Image.fromarray(np.ones(
        (input_size, input_size, 3), dtype=np.uint8)
    )
    x = transform(random_image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
    
    print(outputs.shape)

    summary(
        model, 
        input_data=x,
        col_names=('input_size', 'output_size', 'num_params'),
        row_settings=['var_names'],
    )