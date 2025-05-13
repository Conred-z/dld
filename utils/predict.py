# 预测模块
import torch
from torchvision import transforms
from PIL import Image   # 导入 Python Imaging Library（PIL）中的 Image 模块，用于加载和处理图像。
from models.resnet50 import get_resnet50_model
from config import *

# 定义一个名为 predict_image 的函数，用于对单张图像进行预测。
def predict_image(image_path, model, transform, classes):

    # 打开指定路径的图像文件，返回一个 PIL.Image 对象。
    img = Image.open(image_path)

    # transform 是一个 torchvision.transforms 的组合，通常包括调整图像大小、归一化等操作。
    # 调用 .unsqueeze(0) 在张量的第 0 维添加一个额外的维度，将单张图像的张量形状从 [C, H, W]（通道数、高度、宽度）扩展为 [1, C, H, W]，以适配模型的输入要求。
    img = transform(img).unsqueeze(0).to(DEVICE)

    # 调用 model.eval() 将模型设置为评估模式，这会关闭某些在训练阶段启用的层（如 Dropout、BatchNorm 等）的特定行为，确保模型在推理时表现一致。
    model.eval()

    # 禁用梯度计算并进行前向传播
    with torch.no_grad(): # 使用 torch.no_grad() 上下文管理器，禁用梯度计算，这可以减少内存占用并提高推理速度。

        # 将预处理后的图像张量 img 传递给模型 model，得到模型的输出 output。
        output = model(img)
        # 使用 torch.max(output, 1) 获取模型预测的类别索引 predicted，torch.max 返回每个样本预测概率最大的类别索引。
        _, predicted = torch.max(output, 1)
    # 使用 classes[predicted.item()] 根据预测的类别索引从类别名称列表 classes 中获取对应的类别名称。
    return classes[predicted.item()]