import torch
import torch.nn as nn
from torchvision import models


def load_pretrained_model(model_name, out_ftrs):
    
    all_models = [
        'resnet18',
        'resnet34',
        'squeezenet1_0',
        'mobilenet_v3_large',
        'mobilenet_v3_small',
        'resnext50_32x4d',
        'wide_resnet50_2',
        'wide_resnet101_2',
        'efficientnet_b0'
    ]
    
    assert model_name in all_models, f"model_name is invalid"

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_ftrs)

    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_ftrs)

    elif model_name == 'squeezenet1_0':
        model = models.squeezenet1_0(pretrained=True)
        num_ftrs = model.classifier[1].in_channels
        model.classifier[1] = nn.Conv2d(num_ftrs, out_ftrs, kernel_size=(1, 1), stride=(1, 1))

    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=True)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, out_ftrs)

    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=True)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, out_ftrs)

    elif model_name == 'resnext50_32x4d':
        model = models.resnext50_32x4d(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_ftrs)

    elif model_name == 'wide_resnet50_2':
        model = models.wide_resnet50_2(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_ftrs)

    elif model_name == 'wide_resnet101_2':
        model = models.wide_resnet101_2(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_ftrs)

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, out_ftrs)
        
    return model