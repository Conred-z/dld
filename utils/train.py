# 训练模块
import torch
import torch.nn as nn   # 包含神经网络模块和损失函数
import torch.optim as optim # 包含优化器，如 Adam、SGD 等
from models.resnet50 import get_resnet50_model # 模块中导入 get_resnet50_model 函数，用于获取 ResNet50 模型
from utils.data_loader import get_data_loaders # 
from config import * # 模块中导入所有配置项（假设 config 模块中定义了如 DEVICE 等全局变量）。


############ V3.0 #########################
def train_model(model, criterion, optimizer, dataloaders, num_epochs):
    best_acc = 0.0  # 用于记录最佳验证集准确率

    # 外层循环：遍历每个 epoch
    for epoch in range(num_epochs):
        # 打印当前 epoch 信息
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        # 内层循环：遍历训练和验证阶段
        for phase in ['train', 'val']:
            # 设置模型的训练或评估模式
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # 初始化运行时损失和正确数量
            running_loss = 0.0
            running_corrects = 0

            # 遍历数据加载器中的数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE, dtype=torch.float32)  # 确保输入数据类型为 float32
                labels = labels.to(DEVICE, dtype=torch.int32)  # 确保标签数据类型为 int32

                # 在每次迭代开始时，清零优化器的梯度，避免梯度累积。
                optimizer.zero_grad()

                # 前向传播、计算损失和预测
                with torch.set_grad_enabled(phase == 'train'):  # 使用 torch.set_grad_enabled(phase == 'train') 上下文管理器，仅在“训练阶段”启用梯度计算，验证阶段不计算梯度以节省计算资源。
                    outputs = model(inputs)     # 将输入数据 inputs 传递给模型 model，得到模型的输出 outputs
                    _, preds = torch.max(outputs, 1) # 使用 torch.max(outputs, 1) 获取模型预测的类别（preds），torch.max 返回每个样本预测概率最大的类别索引。
                    loss = criterion(outputs, labels) # 使用损失函数 criterion 计算模型输出 outputs 和真实标签 labels 之间的损失 loss 。

                    # 反向传播和优化（仅在训练阶段）
                    if phase == 'train':
                        loss.backward() # 如果当前阶段是 'train'，调用 loss.backward() 进行反向传播，计算梯度。
                        optimizer.step() # 调用 optimizer.step() 更新模型参数。

                running_loss += loss.item() * inputs.size(0) # 将当前批次的损失 loss.item() 乘以批次大小 inputs.size(0)，累加到 running_loss 中，得到当前阶段的累计损失。
                running_corrects += torch.sum(preds == labels.data) # 计算当前批次中模型预测正确的样本数量 torch.sum(preds == labels.data)，累加到 running_corrects 中。

            # 计算当前阶段(epoch)的平均损失和准确率
            epoch_loss = running_loss / len(dataloaders[phase].dataset) # 计算当前阶段的平均损失 epoch_loss，将累计损失 running_loss 除以当前阶段的数据集大小 len(dataloaders[phase].dataset)。
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset) # 计算当前阶段的准确率 epoch_acc，将累计正确数 running_corrects 转换为浮点数后除以当前阶段的数据集大小。

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存模型逻辑
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'logs/best_model.pth')  # 保存模型权重
                print(f"模型已保存，当前最佳验证集准确率为: {best_acc:.4f}")

    print('训练完成')
    print(f'最佳验证集准确率: {best_acc:.4f}')
    return model

############ V2.0 #########################
# def train_model(model, criterion, optimizer, dataloaders, num_epochs):
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs}')
#         print('-' * 10)

#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             running_corrects = 0

#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(DEVICE, dtype=torch.float32)  # 确保输入数据类型为 float32
#                 labels = labels.to(DEVICE, dtype=torch.int32)  # 确保标签数据类型为 int32

#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / len(dataloaders[phase].dataset)
#             epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#     return model

############# V1.0 #########################
# def train_model(model, criterion, optimizer, dataloaders, num_epochs):
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             running_corrects = 0

#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(DEVICE)
#                 labels = labels.to(DEVICE)

#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / len(dataloaders[phase].dataset)
#             epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 torch.save(model.state_dict(), MODEL_SAVE_PATH)

#     print('Training complete')
#     return model