# 主程序入口
import torch
import torch.nn as nn   # 包含神经网络模块和损失函数
import torch.optim as optim # 包含优化器，如 Adam、SGD 等
from utils.train import train_model # 自定义工具模块，包含训练、评估、预测和数据加载的函数
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
# 定义一个函数 predict_images_in_folder，用于预测指定文件夹中所有图像的类别。
def predict_images_in_folder(model, transform, folder_path, class_names):
    # 初始化两个空列表 predictions 和 true_labels，用于存储预测结果和真实标签。
    predictions = []
    true_labels = []

    # 使用字典推导式创建一个字典 label_map，将类别名称映射到对应的索引。
    # enumerate(class_names) 生成类别名称及其索引，class_name: i 将类别名称作为键，索引作为值。
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
    criterion = nn.CrossEntropyLoss()   # 定义损失函数为 nn.CrossEntropyLoss()，这是分类任务中常用的交叉熵损失函数。

    # 调用 model.load_state_dict 将加载的权重应用到模型上。
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE)) # 参数一：加载的模型权重，参数二：指定加载到哪个设备上（如 CPU 或 GPU）。
    # 将模型设置为评估模式，这会关闭某些在训练阶段启用的层（如 Dropout、BatchNorm 等）的特定行为，确保模型在推理时表现一致。
    model.eval()

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),     # 将像素调整为256*256像素
        transforms.CenterCrop(224), # 使用 transforms.CenterCrop 从调整大小后的图像中裁剪出一个 224x224 像素的中心区域，与ResNet50的输入要求一致
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

    # 使用 np.bincount 统计预测结果 predictions 和真实标签 true_labels 中每个类别的出现次数。
    pred_counts = np.bincount(predictions, minlength=len(class_names))
    # 参数 minlength=len(class_names) 确保统计结果的长度与类别数量一致。
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