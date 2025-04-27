# 模型定义
import torch.nn as nn
import torchvision.models as models

def get_resnet50_model(num_classes):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model