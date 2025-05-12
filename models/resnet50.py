# 模型定义
# import torchvision.models as models

# def get_resnet50_model(num_classes):
#     model = models.resnet50(pretrained=True)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model
import torch.nn as nn
from torchvision import models

def get_resnet50_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model