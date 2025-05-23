import os
import io
import time
import torch
import nibabel as nib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pyzstd
# from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler
import math

# MONAI 中的 SwinUNETR 作为示例网络，需要安装 monai
#   pip install monai
#from monai.networks.nets import SwinUNETR
import importlib.util
spec = importlib.util.spec_from_file_location('SwinUNETR', '/home/wangmo/cluster/slim_unetr/Slim-UNETR/src/SlimUNETR/swin_unetr.py')
my_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(my_module)
SwinUNETR = my_module.SwinUNETR

def custom_collate(batch):
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, dim=0)
    elif isinstance(batch[0], tuple):
        # 如果 batch 中的元素是元组，则分别对每个元素进行堆叠
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]
    else:
        raise TypeError("Unsupported data type in batch.")

# ============ 1) Dataset类 ============
class VolumeDataset(Dataset):
    """
    从 masked_data_dir / original_data_dir 读取对应的 .npy 文件，
    并从 ribbon_mask_dir 中加载 ribbon 掩码 (.nii 或 .nii.gz)。
    
    返回:
        (normalized_masked, normalized_original, background_mask_3d)
        其中:
          - normalized_* 形状: [T, X, Y, Z]
          - background_mask_3d 形状: [1, X, Y, Z], True 表示背景
    """
    def __init__(self, masked_data_dir, original_data_dir, ribbon_mask_dir):
        self.masked_data_dir = masked_data_dir
        self.original_data_dir = original_data_dir
        self.ribbon_mask_dir = ribbon_mask_dir


        self.masked_file_list = sorted(f for f in os.listdir(masked_data_dir) if f.endswith('.zst'))
        self.original_file_list = sorted(f for f in os.listdir(original_data_dir) if f.endswith('.zst'))

        assert len(self.masked_file_list) == len(self.original_file_list), \
            "masked_data 和 original_data 文件数量不一致"

    def __len__(self):
        return len(self.masked_file_list)

    # def __getitem__(self, idx):
    #     masked_name = self.masked_file_list[idx]     # e.g. "3009_rfMRI_REST2_AP_hp2000_1.npy"
    #     original_name = self.original_file_list[idx] # e.g. "3009_rfMRI_REST2_AP_hp2000_1.npy"

    #     masked_path = os.path.join(self.masked_data_dir, masked_name)
    #     original_path = os.path.join(self.original_data_dir, original_name)


    #     masked_volume = load_zstd_file(masked_path)      # shape [X, Y, Z, T]
    #     original_volume = load_zstd_file(original_path)  # shape [X, Y, Z, T]
    #     masked_volume = np.transpose(masked_volume, (3, 0, 1, 2))   # -> [T, X, Y, Z]
    #     original_volume = np.transpose(original_volume, (3, 0, 1, 2)) # -> [T, X, Y, Z]
        
    #     background_mask_3d = original_volume[0]==0
    #     background_mask_3d = background_mask_3d[np.newaxis,:]

        
    #     expanded_bmask = np.broadcast_to(
    #         background_mask_3d, 
    #         original_volume.shape
    #     )  # shape [T, X, Y, Z], True=背景, False=脑组织

    #     # 找到非背景坐标
    #     # non_background = ~background_mask_3d  # shape [T, X, Y, Z], True=脑组织
    #     non_background_values = original_volume[expanded_bmask]

    #     orig_min = np.min(non_background_values)
    #     orig_max = np.max(non_background_values)

    #     # -------- 处理极端情况 (常量体素) -------
    #     if orig_min == orig_max:
    #         # 原始数据在非背景区域是常数，不做归一化处理
    #         normalized_original = original_volume
    #         normalized_masked = masked_volume
    #     else:
    #         # 对 original_volume 非背景区域归一化
    #         normalized_original = np.copy(original_volume)
    #         normalized_original[~expanded_bmask] = (
    #             (original_volume[~expanded_bmask] - orig_min) / (orig_max - orig_min)
    #         )

    #         # 同样对 masked_volume 进行归一化
    #         # 但如果 masked_volume==0 是“被掩掉的值”，你可自行决定是否也视为背景
    #         # 这里只做示例，不区分这两种 0
    #         normalized_masked = np.copy(masked_volume)
    #         nonbg_masked = ~expanded_bmask  # + (masked_volume != 0) 也可以再与 masked_volume!=0 相加
    #         normalized_masked[nonbg_masked] = (
    #             (masked_volume[nonbg_masked] - orig_min) / (orig_max - orig_min)
    #         )

    #     # 转成 torch 张量
    #     norm_masked_torch = torch.tensor(normalized_masked, dtype=torch.float32)
    #     norm_original_torch = torch.tensor(normalized_original, dtype=torch.float32)
    #     bmask_torch = torch.tensor(background_mask_3d, dtype=torch.bool)  # [1, X, Y, Z]

    #     return norm_masked_torch, norm_original_torch, bmask_torch
    def __getitem__(self, idx):
        masked_name = self.masked_file_list[idx]     # e.g. "3009_rfMRI_REST2_AP_hp2000_1.npy"
        original_name = self.original_file_list[idx] # e.g. "3009_rfMRI_REST2_AP_hp2000_1.npy"

        masked_path = os.path.join(self.masked_data_dir, masked_name)
        original_path = os.path.join(self.original_data_dir, original_name)

        masked_volume = load_zstd_file(masked_path)      # shape [X, Y, Z, T]
        original_volume = load_zstd_file(original_path)  # shape [X, Y, Z, T]
        masked_volume = np.transpose(masked_volume, (3, 0, 1, 2))   # -> [T, X, Y, Z]
        original_volume = np.transpose(original_volume, (3, 0, 1, 2)) # -> [T, X, Y, Z]
        
        background_mask_3d = original_volume[0] == 0  # 背景掩码 [X, Y, Z]
        background_mask_3d = background_mask_3d[np.newaxis, :]  # -> [1, X, Y, Z]

        # expanded_bmask = np.broadcast_to(
        #     background_mask_3d, 
        #     original_volume.shape
        # )  # shape [T, X, Y, Z], True=背景, False=脑组织

        # -------- Z-score 归一化 --------
        # 对每个体素的时间序列进行 Z-score 归一化

    # 对 original_volume 和 masked_volume 进行 Z-score 归一化
        normalized_original = zscore_normalize(original_volume, background_mask_3d)
        normalized_masked = zscore_normalize(masked_volume, background_mask_3d)

        # 转成 torch 张量
        norm_masked_torch = torch.tensor(normalized_masked, dtype=torch.float32)
        norm_original_torch = torch.tensor(normalized_original, dtype=torch.float32)
        bmask_torch = torch.tensor(background_mask_3d, dtype=torch.bool)  # [1, X, Y, Z]

        return norm_masked_torch, norm_original_torch, bmask_torch


# ============ 2) 自定义 Loss ============
def masked_mse_loss(pred, target, background_mask_3d):
    """
    pred / target: [B, T, X, Y, Z]
    background_mask_3d: [B, 1, X, Y, Z], True 表示背景不计算Loss。
    
    在这里做广播 -> [B, T, X, Y, Z].
    """
    # pred.shape:   (B, T, X, Y, Z)
    # bmask_3d.shape: (B, 1, X, Y, Z)
    # 扩展到 (B, T, X, Y, Z)
    expanded_mask = background_mask_3d.expand(pred.size(0), pred.size(1), *background_mask_3d.shape[2:])
    # True = 背景，需要过滤掉
    diff = pred - target
    valid_diff = diff[~expanded_mask]  # 只在非背景处计算

    if valid_diff.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return torch.mean(valid_diff**2)

def zscore_normalize(volume, background_mask):
            # 复制数据以避免修改原始数据
            normalized_volume = np.copy(volume)
            
            # 遍历每个体素 (X, Y, Z)
            for x in range(volume.shape[1]):
                for y in range(volume.shape[2]):
                    for z in range(volume.shape[3]):
                        # 如果当前体素不是背景
                        if not background_mask[0, x, y, z]:
                            # 提取时间序列
                            time_series = volume[:, x, y, z]
                            
                            # 计算均值和标准差
                            mean = np.mean(time_series)
                            std = np.std(time_series)
                            
                            # 如果标准差不为零，进行 Z-score 归一化
                            if std != 0:
                                normalized_volume[:, x, y, z] = (time_series - mean) / std
                            else:
                                # 如果标准差为零，设为 0（或根据需要处理）
                                normalized_volume[:, x, y, z] = 0
            
            return normalized_volume
# ============ 3) 训练/验证函数 ============
def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    criterion,
    optimizer,
    device,
    start_epoch=0,
    save_interval=2,
    checkpoint_dir='./checkpoints'
):
    model.train()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        running_loss = 0.0

        for i , (inputs, targets, bmask) in enumerate(train_loader):
            # inputs, targets: [B, T, X, Y, Z]
            # bmask: [B, 1, X, Y, Z]
            inputs = inputs.to(device)
            targets = targets.to(device)
            bmask = bmask.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # [B, T, X, Y, Z]
            loss = criterion(outputs, targets, bmask)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if i % 100 == 99:  # 每 100 个 batch 打印一次
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], "
                    f"Loss: {running_loss / 100:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
                running_loss = 0.0

        # avg_loss = running_loss / len(train_loader)
        duration = time.time() - epoch_start
        print(f"[Train] Epoch {epoch+1}/{num_epochs}, Time: {duration:.2f}s")
        # 验证
        val_loss = validate_model(model, val_loader, criterion, device)
        print(f"[Val]   Epoch {epoch+1}/{num_epochs}, Loss: {val_loss:.4f}")

        # 定期保存检查点
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(checkpoint_dir, f"swin_model_train_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, save_path)

    print("Training finished.")


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets, bmask in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            bmask = bmask.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets, bmask)
            val_loss += loss.item()
    return val_loss / len(val_loader)


# ============ 4) 模型保存与加载 ============
def save_checkpoint(model, optimizer, epoch, file_path):
    state_dict = (
        model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    )
    state = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict()
    }
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(state, file_path)
    print(f"Checkpoint saved: {file_path}")


def load_checkpoint(file_path, model, optimizer):
    if os.path.isfile(file_path):
        print(f"Loading checkpoint '{file_path}'")
        checkpoint = torch.load(file_path)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch+1}")
        return start_epoch
    else:
        print(f"No checkpoint found at '{file_path}', start from scratch.")
        return 0

def load_zstd_file(file_path):
        with open(file_path, 'rb') as f:
            decompressed_data = pyzstd.decompress(f.read())
            buffer = io.BytesIO(decompressed_data)
            return np.load(buffer)
        
class CosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.T_mult == 1:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * self.last_epoch / self.T_0)) / 2
                    for base_lr in self.base_lrs]
        else:
            T_cur = self.last_epoch
            T_i = self.T_0
            while T_cur >= T_i:
                T_cur -= T_i
                T_i *= self.T_mult
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * T_cur / T_i)) / 2
                    for base_lr in self.base_lrs]


if __name__ == "__main__":

    masked_data_dir = "/mnt/dataset1/DATASETS/Chinese_HCP/task/task_masked"
    original_data_dir = "/mnt/dataset1/DATASETS/Chinese_HCP/task/task_origin"
    ribbon_mask_dir = "/mnt/dataset0/DATASETS/Chinese_HCP/CHCP_ribbon_96^3"

    # 一些超参数
    batch_size = 1
    num_epochs = 30
    learning_rate = 1e-3 #1e-3
    checkpoint_dir = "./checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "swin_train_403_4.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建数据集
    dataset = VolumeDataset(masked_data_dir, original_data_dir, ribbon_mask_dir)
    # 划分 90% 训练, 10% 验证
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate)

    # 构建模型 (SwinUNETR 仅作示例)
    model = SwinUNETR(
        img_size=(96, 96, 96),  # 你需要根据实际数据形状设置
        in_channels=4,        # 如果 T=300
        out_channels=4,       # 输出同样 channels
        feature_size=48,
        emb_size=48,
        use_checkpoint=False,
    )

    # 多GPU并行 (可选)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # 优化器
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


    # (可选) 如果有之前的检查点，则加载
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer)


    # 训练
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        criterion=masked_mse_loss,
        optimizer=optimizer,
        device=device,
        start_epoch=start_epoch,
        save_interval=2,  # 每隔多少epoch保存一次
        checkpoint_dir=checkpoint_dir
    )
