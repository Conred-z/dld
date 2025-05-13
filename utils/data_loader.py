# 数据加载器
import os   # 导入操作系统相关的模块，用于处理文件路径等操作
from torchvision import datasets, transforms  # 从 torchvision 库中导入 datasets 和 transforms 模块。datasets 提供了常用的数据集加载器，transforms 提供了数据预处理的方法
from torch.utils.data import DataLoader # 从 torch.utils.data 中导入 DataLoader 类，用于批量加载数据，将数据集封装成可迭代的批次数据。

# 接收四个参数：训练集、验证集、测试集的路径和批量大小
# 返回一个字典，包含训练集、验证集和测试集的 DataLoader
def get_data_loaders(train_dir, val_dir, test_dir, batch_size):
    # 定义数据预处理方法
    data_transforms = {                         # 是一个字典，用于存储不同数据集（训练集、验证集、测试集）的预处理方法。
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪图像到 224×224 的大小，增加了数据的多样性。
            transforms.RandomHorizontalFlip(),  # 随机水平翻转图像，进一步增强数据的多样性。
            transforms.ToTensor(),              # 将图像数据从 PIL 图像格式转换为 PyTorch 的张量格式。
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 对图像进行标准化处理，将每个通道的像素值减去均值并除以标准差，使数据分布更加合理。
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),             # 将图像缩放到 256×256 的大小。
            transforms.CenterCrop(224),         # 从中心裁剪出 224×224 的图像。
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 加载数据集
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']), # 是 PyTorch 提供的一个数据集加载器，它会自动读取指定目录下的图像文件，并根据目录结构来确定图像的类别。
        'val': datasets.ImageFolder(val_dir, data_transforms['val']), # train_dir、val_dir 和 test_dir 分别指定了训练集、验证集和测试集的目录路径。
        'test': datasets.ImageFolder(test_dir, data_transforms['test']) # data_transforms['train']、data_transforms['val'] 和 data_transforms['test'] 分别指定了对应数据集的预处理方法。
    }

    # 创建数据加载器
    # DataLoader 是 PyTorch 提供的一个数据加载器，它可以将数据集分成多个批次，并在训练时按批次加载数据。
    # shuffle=True 表示在每个 epoch 开始时打乱数据顺序，num_workers=4 表示使用 4 个子进程来加载数据，提高数据加载速度。
    # batch_size 是每个批次的大小，通常设置为 32 或 64。
    # 这里创建了三个 DataLoader，分别用于训练集、验证集和测试集。
    # 训练集使用 shuffle=True，验证集和测试集使用 shuffle=False。
    # 这样可以确保在训练时每个 epoch 都能看到不同的训练数据，而在验证和测试时保持数据顺序不变。
    # 这有助于提高模型的泛化能力和评估的准确性。
    # 训练集、验证集和测试集的 DataLoader 分别存储在 image_datasets 字典中。
    # 最后，返回一个包含训练集、验证集和测试集 DataLoader 的字典。
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)
    }

    return dataloaders