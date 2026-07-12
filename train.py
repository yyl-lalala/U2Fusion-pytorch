import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from model import FusionModel
from ewc import EWC
from dataset import H5Dataset

# -------------------- 配置 --------------------
BATCH_SIZE = 18
PATCH_SIZE = 64
LR = 0.0001
EPOCHS = [3, 2, 2]           # 三个任务的轮数
C_VALUES = [3200, 3500, 100] # 信息保留度常数
LAM = 0                  # EWC 惩罚强度
NUM_FISHER = 1000            # 每个任务用于估算 Fisher 的批次数（建议 ≥ 1000）
LOG_INTERVAL = 25
SAVE_INTERVAL = 100
VAL_SIZE = 2000              # 每个任务验证集样本数（论文固定值）
SEED = 42                    # 划分验证集的随机种子，保证可复现

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# -------------------- 数据路径 --------------------
data1_path = './data/vis_ir_dataset64.h5'
data2_path = './data/oe_ue_Y_dataset64.h5'
data3_path = './data/far_near_Y_dataset64.h5'
vgg_weights = 'vgg16.pth'

# -------------------- 完整数据集 --------------------
full_set1 = H5Dataset(data1_path)
full_set2 = H5Dataset(data2_path)
full_set3 = H5Dataset(data3_path)

# -------------------- 划分训练/验证集 --------------------
def split_dataset(dataset, val_size=VAL_SIZE, seed=SEED):
    indices = list(range(len(dataset)))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    if val_size > len(dataset):
        raise ValueError(f'VAL_SIZE ({val_size}) exceeds dataset size ({len(dataset)})')
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

train_set1, val_set1 = split_dataset(full_set1)
train_set2, val_set2 = split_dataset(full_set2)
train_set3, val_set3 = split_dataset(full_set3)

print(f'Task1: train={len(train_set1)}, val={len(val_set1)}')
print(f'Task2: train={len(train_set2)}, val={len(val_set2)}')
print(f'Task3: train={len(train_set3)}, val={len(val_set3)}')

# -------------------- DataLoader --------------------
def collate_random_channels(batch):
    imgs = torch.stack(batch, dim=0)
    idx = torch.randint(0, 2, (1,)).item()
    S1 = imgs[:, idx:idx+1, :, :]
    S2 = imgs[:, 1-idx:1-idx+1, :, :]
    return {'S1': S1, 'S2': S2}

def collate_fixed_channels(batch):
    imgs = torch.stack(batch, dim=0)
    S1 = imgs[:, 0:1, :, :]
    S2 = imgs[:, 1:2, :, :]
    return {'S1': S1, 'S2': S2}

train_loader1 = DataLoader(train_set1, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_random_channels, drop_last=True)
train_loader2 = DataLoader(train_set2, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_random_channels, drop_last=True)
train_loader3 = DataLoader(train_set3, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_random_channels, drop_last=True)

val_loader1 = DataLoader(val_set1, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate_fixed_channels)
val_loader2 = DataLoader(val_set2, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate_fixed_channels)
val_loader3 = DataLoader(val_set3, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate_fixed_channels)

# -------------------- 模型、优化器 --------------------
model = FusionModel(vgg_weights_path=vgg_weights).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=LR, momentum=0.15, alpha=0.9)

# -------------------- 工具函数 --------------------
@torch.no_grad()
def evaluate_loss(model, dataloader, c_value):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    for batch in dataloader:
        S1 = batch['S1'].to(device)
        S2 = batch['S2'].to(device)
        fused = model(S1, S2)
        loss = model.content_loss(S1, S2, fused, c_value)
        total_loss += loss.item() * S1.size(0)
        n_samples += S1.size(0)
    return total_loss / n_samples if n_samples > 0 else 0.0

def compute_total_ewc_loss(ewc_list, lam):
    total = 0.0
    for ewc in ewc_list:
        total = total + ewc.ewc_loss(lam)
    return total

def train_epoch(loader, model, optimizer, epoch, c_value,
                ewc_list=None, lam=0, writer=None,
                val_writer=None,
                val_loaders_for_eval=None,   # {task_name: (loader, c_value)}
                global_step=0):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        S1, S2 = batch['S1'].to(device), batch['S2'].to(device)
        optimizer.zero_grad()
        fused = model(S1, S2)
        loss_content = model.content_loss(S1, S2, fused, c_value)

        if ewc_list:
            loss = loss_content + compute_total_ewc_loss(ewc_list, lam)
        else:
            loss = loss_content

        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 50)
        optimizer.step()

        total_loss += loss.item()
        if writer and batch_idx % LOG_INTERVAL == 0:
            writer.add_scalar('loss/total', loss.item(), global_step)
            writer.add_scalar('loss/content', loss_content.item(), global_step)

        # ---------- 定期评估旧任务验证损失（统一父标签，便于同图显示） ----------
        if val_writer and val_loaders_for_eval and (batch_idx % LOG_INTERVAL == 0):
            for task_name, (val_loader, c_val) in val_loaders_for_eval.items():
                val_loss = evaluate_loss(model, val_loader, c_val)
                val_writer.add_scalar(f'val_loss/curves/{task_name}', val_loss, global_step)

        global_step += 1

        if batch_idx % LOG_INTERVAL == 0:
            print(f'  Epoch {epoch} Batch {batch_idx}/{len(loader)} Loss: {loss.item():.4f}')
    return total_loss / len(loader), global_step

def save_model(model, path):
    torch.save(model.state_dict(), path)

# ==================== 统一验证记录器 ====================
os.makedirs('logs', exist_ok=True)
val_writer = SummaryWriter('logs/val_curves')

# ★★★ 强制三条曲线叠加在同一张图表中 ★★★
val_writer.add_custom_scalars({
    "Validation Loss": {
        "All Tasks": ["Multiline", [
            "val_loss/curves/task1",
            "val_loss/curves/task2",
            "val_loss/curves/task3"
        ]]
    }
})

# ==================== 任务1：多模图像融合 ====================
print('===== Task 1 (Multi-modal) =====')
os.makedirs('checkpoints/task1', exist_ok=True)
writer_task1 = SummaryWriter('logs/task1')
global_step = 0

val_eval_task1 = {'task1': (val_loader1, C_VALUES[0])}

for epoch in range(1, EPOCHS[0] + 1):
    avg_loss, global_step = train_epoch(train_loader1, model, optimizer, epoch,
                                        C_VALUES[0],
                                        writer=writer_task1,
                                        val_writer=val_writer,
                                        val_loaders_for_eval=val_eval_task1,
                                        global_step=global_step)
    print(f'Task1 Epoch {epoch} Avg Loss: {avg_loss:.4f}')
    save_model(model, f'checkpoints/task1/model_epoch{epoch}_{global_step}.pth')

save_model(model, 'checkpoints/task1/final.pth')
writer_task1.close()

# 过渡点：任务1结束后立即评估
loss_task1_after = evaluate_loss(model, val_loader1, C_VALUES[0])
val_writer.add_scalar('val_loss/curves/task1', loss_task1_after, global_step)

# 计算 Fisher + 创建 EWC
print('Computing Fisher for task1 data...')
ewc1 = EWC(model, device)
def fisher_loss_fn1(S1, S2):
    fused = model(S1, S2)
    return model.content_loss(S1, S2, fused, C_VALUES[0])
ewc1.compute_fisher(train_loader1, fisher_loss_fn1, num_samples=NUM_FISHER)
ewc1.update_star()
ewc_list = [ewc1]

# ==================== 任务2：多曝光图像融合 ====================
print('===== Task 2 (Multi-exposure) =====')
os.makedirs('checkpoints/task2', exist_ok=True)
writer_task2 = SummaryWriter('logs/task2')

val_eval_task2 = {
    'task1': (val_loader1, C_VALUES[0]),
    'task2': (val_loader2, C_VALUES[1])
}

for epoch in range(1, EPOCHS[1] + 1):
    avg_loss, global_step = train_epoch(train_loader2, model, optimizer, epoch,
                                        C_VALUES[1],
                                        ewc_list=ewc_list, lam=LAM,
                                        writer=writer_task2,
                                        val_writer=val_writer,
                                        val_loaders_for_eval=val_eval_task2,
                                        global_step=global_step)
    print(f'Task2 Epoch {epoch} Avg Loss: {avg_loss:.4f}')
    save_model(model, f'checkpoints/task2/model_epoch{epoch}_{global_step}.pth')

save_model(model, 'checkpoints/task2/final.pth')
writer_task2.close()

# 过渡点：任务2结束后立即评估
loss_task1_after2 = evaluate_loss(model, val_loader1, C_VALUES[0])
loss_task2_after2 = evaluate_loss(model, val_loader2, C_VALUES[1])
val_writer.add_scalar('val_loss/curves/task1', loss_task1_after2, global_step)
val_writer.add_scalar('val_loss/curves/task2', loss_task2_after2, global_step)

# 为任务2创建 EWC 保护对象
print('Computing Fisher for task2 data...')
ewc2 = EWC(model, device)
def fisher_loss_fn2(S1, S2):
    fused = model(S1, S2)
    return model.content_loss(S1, S2, fused, C_VALUES[1])
ewc2.compute_fisher(train_loader2, fisher_loss_fn2, num_samples=NUM_FISHER)
ewc2.update_star()
ewc_list.append(ewc2)

# ==================== 任务3：多聚焦图像融合 ====================
print('===== Task 3 (Multi-focus) =====')
os.makedirs('checkpoints/task3', exist_ok=True)
writer_task3 = SummaryWriter('logs/task3')

val_eval_task3 = {
    'task1': (val_loader1, C_VALUES[0]),
    'task2': (val_loader2, C_VALUES[1]),
    'task3': (val_loader3, C_VALUES[2])
}

for epoch in range(1, EPOCHS[2] + 1):
    avg_loss, global_step = train_epoch(train_loader3, model, optimizer, epoch,
                                        C_VALUES[2],
                                        ewc_list=ewc_list, lam=LAM,
                                        writer=writer_task3,
                                        val_writer=val_writer,
                                        val_loaders_for_eval=val_eval_task3,
                                        global_step=global_step)
    print(f'Task3 Epoch {epoch} Avg Loss: {avg_loss:.4f}')
    save_model(model, f'checkpoints/task3/model_epoch{epoch}_{global_step}.pth')

save_model(model, 'checkpoints/task3/final.pth')
writer_task3.close()
val_writer.close()

print('Training Finished. All models saved.')
print('TensorBoard val curves (same graph): tensorboard --logdir=logs/val_curves')