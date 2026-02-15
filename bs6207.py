import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
import copy


# 1. 基础配置 (Configuration)
DATA_DIR = '/Users/xinyue/Documents/Data-Prostate Cancer Patch Classification' 

BATCH_SIZE = 32
LEARNING_RATE = 0.0001  # 使用较小的学习率进行微调 (Fine-tuning)
NUM_EPOCHS = 20         # 训练轮数，建议至少 15-20 轮以观察收敛
NUM_CLASSES = 5         # Stroma, Normal, G3, G4, G5 [cite: 19-24]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Project: Prostate Cancer Patch Classification")
print(f"Using device: {DEVICE}")

# 2. 数据预处理与增强 (Transforms)
# 训练集：需要数据增强来提高泛化能力 (Generalization)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),        # ResNet 标准输入
    transforms.RandomHorizontalFlip(),    # 随机水平翻转
    transforms.RandomVerticalFlip(),      # 随机垂直翻转
    transforms.RandomRotation(90),        # 随机旋转 90 度
    # 颜色抖动：模拟不同染色切片的颜色差异，这对病理图像至关重要
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 验证集和测试集：只需要标准化，不要做随机增强
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. 数据加载与合并 (Data Loading)
# ==========================================
print("\n--- Loading Datasets ---")

# 定义具体的文件夹路径 (根据你提供的结构)
path_train_a = os.path.join(DATA_DIR, 'SetA', '251_Train_A')
path_test_a  = os.path.join(DATA_DIR, 'SetA', '251_Test_A')
path_train_b = os.path.join(DATA_DIR, 'SetB', '251_Train_B')
path_test_b  = os.path.join(DATA_DIR, 'SetB', '251_Test_B')
path_ntu_add = os.path.join(DATA_DIR, 'Test_NTU_additional')

# 检查路径是否存在
for p in [path_train_a, path_test_a, path_train_b, path_test_b, path_ntu_add]:
    if not os.path.exists(p):
        print(f"Error: Path not found: {p}")
        exit()

# 加载各个子数据集
# 注意：SetA 和 SetB 的 Train 部分都用了 train_transforms
ds_train_a = datasets.ImageFolder(path_train_a, transform=train_transforms)
ds_train_b = datasets.ImageFolder(path_train_b, transform=train_transforms)

# 注意：SetA 和 SetB 的 Test 部分在这里充当 "验证集"，使用 val_test_transforms
ds_val_a   = datasets.ImageFolder(path_test_a, transform=val_test_transforms)
ds_val_b   = datasets.ImageFolder(path_test_b, transform=val_test_transforms)

# 最终测试集
ds_test_final = datasets.ImageFolder(path_ntu_add, transform=val_test_transforms)

# --- 关键步骤：合并数据集 ---
# 1. 训练集 = Train_A + Train_B
full_train_dataset = ConcatDataset([ds_train_a, ds_train_b])

# 2. 验证集 = Test_A + Test_B (用于调参)
full_val_dataset = ConcatDataset([ds_val_a, ds_val_b])

# 3. 最终测试集
final_test_dataset = ds_test_final

# 打印类别映射关系 (重要！检查 G3, G4, G5, Normal, Stroma 的顺序)
class_names = ds_train_a.classes
print(f"Class Mapping: {ds_train_a.class_to_idx}")
print(f"Training Samples: {len(full_train_dataset)} (SetA Train + SetB Train)")
print(f"Validation Samples: {len(full_val_dataset)} (SetA Test + SetB Test)")
print(f"Final Test Samples: {len(final_test_dataset)} (NTU Additional)")

# 创建 DataLoaders
# num_workers 在 Windows 下如果报错请改为 0
train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(full_val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(final_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ==========================================
# 4. 处理数据不平衡 (Imbalance Handling)
# ==========================================
print("\n--- Handling Class Imbalance ---")
# 这是一个加分项 (Address challenges associated with imbalanced dataset)
# 我们需要统计合并后的训练集里每个类有多少张图

# 因为 ConcatDataset 没有 targets 属性，我们需要手动拼接
all_train_targets = []
for ds in full_train_dataset.datasets:
    all_train_targets.extend(ds.targets)

class_counts = np.bincount(all_train_targets)
print(f"Class Counts in Training: {dict(zip(class_names, class_counts))}")

# 计算 Loss 权重: 样本越少，权重越大
# Weight = Total / (Num_Classes * Count)
total_samples = len(all_train_targets)
class_weights = total_samples / (NUM_CLASSES * class_counts)
# 转为 Tensor 并移到 GPU/CPU
class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
print(f"Computed Class Weights: {class_weights}")

# ==========================================
# 5. 模型构建 (Model Setup)
# ==========================================
# 使用 ResNet18 进行迁移学习 [cite: 7]
model = models.resnet18(pretrained=True)

# 修改最后一层全连接层 (FC) 以匹配 5 个类别
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(DEVICE)

# 定义 Loss Function (传入 class_weights 解决不平衡)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# 定义优化器 (Adam 通常收敛更快)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 6. 训练循环 (Training Loop)
# ==========================================
def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 记录数据用于画图 (Optimization Analysis)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每个 Epoch 分为训练和验证两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式 (启用 Dropout, BatchNorm 更新)
                dataloader = train_loader
            else:
                model.eval()   # 评估模式
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播 (只在训练阶段)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计 Loss 和 准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 保存历史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

                # 如果验证集准确率是历史最高，保存这个模型
                # 这是 "Optimization" 的关键，防止过拟合
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Acc: {best_acc:.4f}')

    # 加载验证集表现最好的模型权重作为最终模型
    model.load_state_dict(best_model_wts)
    return model, history

# 开始训练
print("\n--- Starting Training ---")
trained_model, history = train_model(model, criterion, optimizer, NUM_EPOCHS)

# ==========================================
# 7. 结果可视化 (Visualization)
# ==========================================
# 这两张图是报告中 Discussion 部分关于 "Optimization" 的核心证据
plt.figure(figsize=(12, 5))

# Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss vs. Epochs (Optimization)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title('Accuracy vs. Epochs (Generalization Check)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# 8. 最终测试 (Final Evaluation)
# ==========================================
print("\n--- Final Evaluation on Test_NTU_additional ---")

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_preds

# 在 NTU_additional 上预测
y_true, y_pred = evaluate(trained_model, test_loader)

# 打印详细报告 (包含 Precision, Recall, F1) 
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# 绘制混淆矩阵 (Confusion Matrix)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix - Final Test Set')
plt.show()

# 保存模型
torch.save(trained_model.state_dict(), 'prostate_cancer_final_model.pth')
print("\nModel saved successfully.")