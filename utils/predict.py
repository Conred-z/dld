# 预测模块
import torch
from torchvision import transforms
from PIL import Image
from models.resnet50 import get_resnet50_model
from config import *

def predict_image(image_path, model, transform, classes):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    return classes[predicted.item()]