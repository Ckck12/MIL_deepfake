# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# import datetime
# from dataset import get_dataloader
# from model import FrameFeatureExtractor, SMILModel, smil_loss


# def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, device):
#     if not os.path.exists('checkpoint'):
#         os.makedirs('checkpoint')
    
#     best_val_loss = float('inf')
#     writer = SummaryWriter()  # For TensorBoard logging

#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         total = 0
#         correct = 0
        
#         # Training loop
#         for videos, labels, _ in tqdm(train_loader, desc=f"에포크 [{epoch+1}/{num_epochs}] 학습 진행", dynamic_ncols=True):
#             videos, labels = videos.to(device), labels.to(device).float()
#             # videos: B x 3F x H x W - > BF x 3 x H x W
#             optimizer.zero_grad()
#             try:
#                 # print(videos.shape)
#                 outputs, _ = model(videos) # BF x 1 -> B x F x 1
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
#                 train_loss += loss.item() 
#                 predicted = (outputs > 0.5).float()
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#             except ValueError as e:
#                 print(f"Skipping batch due to error: {e}")
#                 continue

#         train_loss /= len(train_loader.dataset)
#         writer.add_scalar('Loss/train', train_loss, epoch)
#         train_acc = correct / total
        
#         val_loss, val_acc, _ = validate_model(val_loader, model, criterion, device)
#         writer.add_scalar('Loss/val', val_loss, epoch)
#         writer.add_scalar('Accuracy/val', val_acc, epoch)

#         print(f"에포크 [{epoch+1}/{num_epochs}] - train loss: {train_loss:.4f}, train acc: {train_acc:.4f} val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             now = datetime.datetime.now()
#             # Format as string
#             now_str = now.strftime('%Y%m%d_%H%M%S')
#             # Save the best model name with date and time
#             torch.save(model.state_dict(), f'checkpoint/best_model_{now_str}.pth')
        
#         if (epoch + 1) % 5 == 0:
#             torch.save(model.state_dict(), f'checkpoint/epoch_{epoch+1}_{now_str}.pth')

#     writer.close()

# def validate_model(val_loader, model, criterion, device):
#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0
#     video_names = []

#     with torch.no_grad():
#         for videos, labels, video_name in tqdm(val_loader, desc="검증 진행", dynamic_ncols=True):
#             videos, labels = videos.to(device), labels.to(device).float()
#             try:
#                 outputs, _ = model(videos)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item() 
#                 predicted = (outputs > 0.5).float()
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#                 video_names.extend(video_name)
#             except ValueError as e:
#                 print(f"Skipping batch due to error: {e}")
#                 continue

#     val_loss /= len(val_loader.dataset)
#     val_acc = correct / total
#     return val_loss, val_acc, video_names

# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
from dataset import get_dataloader
from model import FrameFeatureExtractor, SMILModel, smil_loss
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, device, scaler):
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    
    best_val_loss = float('inf')
    writer = SummaryWriter()  # For TensorBoard logging

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total = 0
        correct = 0

        # Set the epoch for the sampler
        train_loader.sampler.set_epoch(epoch)
        
        # Training loop
        for videos, labels, _ in tqdm(train_loader, desc=f"에포크 [{epoch+1}/{num_epochs}] 학습 진행", dynamic_ncols=True):
            videos, labels = videos.to(device), labels.to(device).float()
            optimizer.zero_grad()
            with autocast():
                outputs, _ = model(videos)
                weight_real = 1 / 0.75
                weight_fake = 1 / 0.25
                weights = weight_real if labels.mean() > 0.5 else weight_fake
                loss = criterion(outputs, labels, weights)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() 
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.sampler)
        writer.add_scalar('Loss/train', train_loss, epoch)
        train_acc = correct / total
        
        val_loss, val_acc, _ = validate_model(val_loader, model, criterion, device)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"에포크 [{epoch+1}/{num_epochs}] - train loss: {train_loss:.4f}, train acc: {train_acc:.4f} val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            now = datetime.datetime.now()
            now_str = now.strftime('%Y%m%d_%H%M%S')
            if dist.get_rank() == 0:  # Only save on the main process
                torch.save(model.state_dict(), f'checkpoint/best_model_{now_str}.pth')
        
        if (epoch + 1) % 5 == 0 and dist.get_rank() == 0:
            torch.save(model.state_dict(), f'checkpoint/epoch_{epoch+1}_{now_str}.pth')

    writer.close()

def validate_model(val_loader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    video_names = []

    with torch.no_grad():
        for videos, labels, video_name in tqdm(val_loader, desc="검증 진행", dynamic_ncols=True):
            videos, labels = videos.to(device), labels.to(device).float()
            try:
                with autocast():
                    outputs, _ = model(videos)
                    weight_real = 1 / 0.75
                    weight_fake = 1 / 0.25
                    weights = weight_real if labels.mean() > 0.5 else weight_fake
                    loss = criterion(outputs, labels, weights)
                val_loss += loss.item() 
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                video_names.extend(video_name)
            except ValueError as e:
                print(f"Skipping batch due to error: {e}")
                continue

    val_loss /= len(val_loader.sampler)
    val_acc = correct / total
    return val_loss, val_acc, video_names

