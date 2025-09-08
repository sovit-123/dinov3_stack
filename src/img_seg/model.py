import torch
import torch.nn as nn

from torchinfo import summary

model_feature_layers = {
    'dinov3_vits16': [3, 5, 7, 11],
    'dinov3_vits16plus': [3, 5, 7, 11],
    'dinov3_vitb16': [3, 5, 7, 11],
    'dinov3_vitl16': [7, 11, 15, 23],
    'dinov3_vith16plus': [9, 13, 18, 26],
    'dinov3_vit7b16': [11, 16, 21, 31]
}


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
        repo_dir: str=None,
        feature_extractor: str='last' # OR 'multi'
    ):
        super(Dinov3Segmentation, self).__init__()

        self.model_name = model_name

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

        self.feature_extractor_layers = 1 if feature_extractor == 'last' else model_feature_layers[self.model_name]
        decode_head_in_channels = self.backbone_model.norm.normalized_shape[0] if feature_extractor == 'last' else self.backbone_model.norm.normalized_shape[0] * 4

        self.decode_head = SimpleDecoder(
            in_channels=decode_head_in_channels, 
            nc=self.num_classes
        )

    def forward(self, x):
        # Backbone forward pass
        features = self.backbone_model.get_intermediate_layers(
            x, 
            n=self.feature_extractor_layers, 
            reshape=True, 
            return_class_token=False, 
            norm=True
        )

        # for i, feat in enumerate(features):
        #     print(f"Feature {i}: {feat.shape}")
            
        concatednated_features = torch.cat(features, dim=1)

        # print('Final feature shape: ', concatednated_features.shape)
        # exit(0)

        # Decoder forward pass
        classifier_out = self.decode_head(concatednated_features)
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

    model_names = {
        'dinov3_vits16': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'dinov3_vits16plus': 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
        'dinov3_vitb16': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        'dinov3_vitl16': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        'dinov3_vith16plus': 'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
        # 'dinov3_vit7b16': [11, 16, 21, 31]
    }

    for model_name in model_names:
        print('Testing: ', model_name)
        model = Dinov3Segmentation(
            repo_dir=DINOV3_REPO, 
            weights=os.path.join(DINOV3_WEIGHTS, model_names[model_name]),
            model_name=model_name,
            feature_extractor='multi' # OR 'last'
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
        print('#' * 50, '\n\n')