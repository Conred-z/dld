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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


######### V2.0 #########################
def predict_images_in_folder(model, transform, folder_path, class_names):
    predictions = []
    true_labels = []
    label_map = {class_name: i for i, class_name in enumerate(class_names)}
    for class_folder in os.listdir(folder_path):
        class_folder_path = os.path.join(folder_path, class_folder)
        if not os.path.isdir(class_folder_path):
            continue
        for image_name in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_name)
            true_label = label_map[class_folder]
            predicted_class = predict_image(image_path, model, transform, class_names)
            predictions.append(label_map[predicted_class])
            true_labels.append(true_label)
    return predictions, true_labels

######### V1.0 #########################
# def predict_images_in_folder(model, transform, folder_path, classes):
#     predictions = []
#     for class_folder in os.listdir(folder_path):
#         class_folder_path = os.path.join(folder_path, class_folder)
#         if not os.path.isdir(class_folder_path):
#             continue
#         for image_name in os.listdir(class_folder_path):
#             image_path = os.path.join(class_folder_path, image_name)
#             predicted_class = predict_image(image_path, model, transform, classes)
#             predictions.append((image_name, predicted_class))
#     return predictions


if __name__ == '__main__':

############### V2.0 数据可视化部分 #########################
    # 初始化模型、损失函数和优化器
    model = get_resnet50_model(NUM_CLASSES)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # 加载最佳模型权重
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 预测验证集图片
    test_dir = os.path.join(DATA_DIR, 'test')
    class_names = sorted(os.listdir(test_dir))  # 假设类别名是文件夹名
    predictions, true_labels = predict_images_in_folder(model, transform, test_dir, class_names)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(true_labels, predictions)

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # 可视化预测结果的分布
    pred_counts = np.bincount(predictions, minlength=len(class_names))
    true_counts = np.bincount(true_labels, minlength=len(class_names))

    plt.figure(figsize=(12, 6))
    plt.bar(class_names, pred_counts, label='Predicted Counts')
    plt.bar(class_names, true_counts, alpha=0.5, label='True Counts')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution of Predictions and True Labels')
    plt.legend()
    plt.show()

################ V1.0 数据展示部分 ######################
    # # 初始化模型、损失函数和优化器
    # model = get_resnet50_model(NUM_CLASSES)
    # model = model.to(DEVICE)
    # criterion = nn.CrossEntropyLoss()

    # # 加载最佳模型权重
    # model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    # model.eval()

    # # 定义图像预处理
    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    # # 预测验证集图片
    # test_dir = os.path.join(DATA_DIR, 'test')
    # classes = sorted(os.listdir(test_dir))  # 假设类别名是文件夹名
    # predictions = predict_images_in_folder(model, transform, test_dir, classes)

    # # 打印预测结果
    # for image_name, predicted_class in predictions:
    #     print(f'Image: {image_name}, Predicted class: {predicted_class}')

###########################################################
###################### 模型训练部分 #########################
    # # 加载数据
    # dataloaders = get_data_loaders(TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE)

    # # 初始化模型、损失函数和优化器
    # model = get_resnet50_model(NUM_CLASSES)
    # model = model.to(DEVICE)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # # 训练模型
    # model = train_model(model, criterion, optimizer, dataloaders, NUM_EPOCHS)

    # # 加载最佳模型权重
    # model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    # model = model.to(DEVICE)

    # # 评估模型
    # labels, preds = evaluate_model(model, dataloaders['test'])
    # plot_confusion_matrix(labels, preds, dataloaders['test'].dataset.classes)

    # # 测试图片预测
    # test_image_path = 'data/test/0/image_06736.jpg'  # 替换为测试图片路径
    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # predicted_class = predict_image(test_image_path, model, transform, dataloaders['test'].dataset.classes)
    # print(f'Predicted class: {predicted_class}')