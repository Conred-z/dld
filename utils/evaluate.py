# 评估模块
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.data_loader import get_data_loaders
from models.resnet50 import get_resnet50_model
from config import *

# 定义一个函数 evaluate_model，它接受两个参数：model（要评估的模型）和 dataloader（数据加载器，用于加载测试数据）。
def evaluate_model(model, dataloader):
    # 将模型设置为评估模式。在评估模式下，模型的行为会与训练模式有所不同，例如关闭 Dropout 和 Batch Normalization 等层的训练特性。
    model.eval()
    # 初始化两个空列表 all_preds 和 all_labels，分别用于存储所有预测结果和真实标签。
    all_preds = []
    all_labels = []

    # 使用 torch.no_grad() 上下文管理器，表示在该代码块中不计算梯度。因为在评估模型时不需要进行反向传播，这可以减少内存占用并提高计算速度。
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)

            # 使用 torch.max(outputs, 1) 获取模型输出的最大值及其索引。_ 表示忽略最大值，preds 是最大值对应的索引，即模型的预测类别。
            _, preds = torch.max(outputs, 1)

            # 将预测结果和真实标签从 GPU（如果有）移动到 CPU，并转换为 NumPy 数组，然后分别添加到 all_preds 和 all_labels 列表中。
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

def plot_confusion_matrix(labels, preds, classes):
    # 使用 confusion_matrix 函数计算混淆矩阵。混淆矩阵是一个二维数组，其中第 i 行第 j 列的值表示实际为类别 i 而被预测为类别 j 的样本数量。
    cm = confusion_matrix(labels, preds)

    """
    创建一个 ConfusionMatrixDisplay 对象，用于显示混淆矩阵。
    confusion_matrix 参数传入混淆矩阵，display_labels 参数传入类别名称列表，用于在混淆矩阵中显示类别标签。
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    """
    ConfusionMatrixDisplay 可以将混淆矩阵以图表的形式展示出来，通常是一个热力图（heatmap）。
    这种可视化方式使得混淆矩阵中的数据更加直观易懂。
    例如，我们可以很容易地看到哪些类别被正确分类，哪些类别被错误分类，以及错误分类的具体情况。
    """

    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()