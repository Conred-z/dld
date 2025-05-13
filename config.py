import os
import torch

# 数据集路径
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# 模型保存路径
MODEL_SAVE_PATH = 'logs/best_model.pth'

# 超参数
NUM_CLASSES = 5          # 花卉类别数
BATCH_SIZE = 64          # 批次大小
NUM_EPOCHS = 25          # 训练轮次
LEARNING_RATE = 0.001    # 学习旅
WEIGHT_INIT = 'kaiming'  # 权重初始化方法
OPTIMIZER_TYPE = 'adam'  # 优化器类型
MOMENTUM = 0.9           # 动量
LR_SCHEDULER = 'step'    # 学习率调度器
WARMUP_STEPS = 1000      # 学习率预热步数
LR_DECAY = 0.1           # 学习率衰减
WEIGHT_DECAY = 0.0001    # 权重衰减
ACCUMULATION_STEPS = 1   # 梯度累积步数
EARLY_STOPPING = 5       # 早停
DATA_AUGMENTATION = 'random_crop'  # 数据增强策略
LOSS_FUNCTION = 'cross_entropy'  # 损失函数类型

# NUM_CLASSES = 5  # 花卉类别数
# BATCH_SIZE = 64
# NUM_EPOCHS = 25
# LEARNING_RATE = 0.001

# 检查设备
if torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# # 配置文件
# import os
# import torch

# # 数据集路径
# DATA_DIR = 'data'
# TRAIN_DIR = os.path.join(DATA_DIR, 'train')
# VAL_DIR = os.path.join(DATA_DIR, 'val')
# TEST_DIR = os.path.join(DATA_DIR, 'test')

# # 模型保存路径
# MODEL_SAVE_PATH = 'logs/best_model.pth'

# # 超参数
# NUM_CLASSES = 5  # 花卉类别数
# BATCH_SIZE = 32
# NUM_EPOCHS = 25
# LEARNING_RATE = 0.001
# DEVICE = 'mps' if torch.mps.is_available() else 'cpu'