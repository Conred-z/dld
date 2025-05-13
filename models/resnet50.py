# 模型定义
# import torchvision.models as models

# def get_resnet50_model(num_classes):
#     model = models.resnet50(pretrained=True)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model

import torch.nn as nn # 包含神经网络模块和损失函数
from torchvision import models # 从 torchvision 库中导入 models 模块，torchvision 是 PyTorch 的一个扩展库，提供了许多预训练的模型和数据集。

# 定义一个函数 get_resnet50_model，接收一个参数 num_classes，表示分类的类别数
# 该函数返回一个 ResNet50 模型，最后一层的输出节点数为 num_classes
# 该函数使用 torchvision 库中的预训练模型 ResNet50，并将最后一层的输出节点数修改为 num_classes
def get_resnet50_model(num_classes): # 参数 num_classes 表示目标分类任务的类别数，即模型输出层的神经元数量。

    # 参数 weights=models.ResNet50_Weights.IMAGENET1K_V1 指定了使用在 ImageNet 数据集上预训练的权重版本 IMAGENET1K_V1。
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # model.fc 是 ResNet50 模型中的全连接层（fc 表示 fully connected）。
    # model.fc.in_features 获取该全连接层的输入特征数量，即 ResNet50 的最后一个卷积层输出的特征数量。
    # 这个值通常用于确定新全连接层的输入维度。
    num_ftrs = model.fc.in_features

    # 使用 nn.Linear 创建一个新的全连接层，输入特征数量为 num_ftrs，输出特征数量为 num_classes。
    # 将模型中原有的全连接层替换为这个新创建的全连接层。
    # 这样，模型的输出层就适配了新的分类任务，可以输出 num_classes 个类别的预测概率。
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 返回修改后的 ResNet50 模型，该模型在保留了预训练的特征提取部分的同时，将输出层调整为适合当前分类任务的结构。
    return model