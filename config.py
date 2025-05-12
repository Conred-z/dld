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
NUM_CLASSES = 5  # 花卉类别数
BATCH_SIZE = 16
NUM_EPOCHS = 2
LEARNING_RATE = 0.001

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