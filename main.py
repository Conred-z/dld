# 主程序入口
import torch
import torch.nn as nn
import torch.optim as optim
from utils.train import train_model
from utils.evaluate import evaluate_model, plot_confusion_matrix
from utils.predict import predict_image
from utils.data_loader import get_data_loaders
from models.resnet50 import get_resnet50_model
from torchvision import transforms
from config import *

if __name__ == '__main__':
    # 加载数据
    dataloaders = get_data_loaders(TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE)

    # 初始化模型、损失函数和优化器
    model = get_resnet50_model(NUM_CLASSES)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练模型
    model = train_model(model, criterion, optimizer, dataloaders, NUM_EPOCHS)

    # 加载最佳模型权重
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model = model.to(DEVICE)

    # 评估模型
    labels, preds = evaluate_model(model, dataloaders['test'])
    plot_confusion_matrix(labels, preds, dataloaders['test'].dataset.classes)

    # 测试图片预测
    test_image_path = 'test_images/sunflower.jpg'  # 替换为测试图片路径
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    predicted_class = predict_image(test_image_path, model, transform, dataloaders['test'].dataset.classes)
    print(f'Predicted class: {predicted_class}')