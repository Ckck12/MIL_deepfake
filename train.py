import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, device):
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    
    best_val_loss = float('inf')
    writer = SummaryWriter()  # For TensorBoard logging

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Training loop
        for frames, labels, _ in tqdm(train_loader, desc=f"에포크 [{epoch+1}/{num_epochs}] 학습 진행", dynamic_ncols=True):
            frames, labels = frames.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * frames.size(0)

        train_loss /= len(train_loader.dataset)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, labels, _ in tqdm(val_loader, desc=f"에포크 [{epoch+1}/{num_epochs}] 검증 진행", dynamic_ncols=True):
                frames, labels = frames.to(device), labels.to(device).float()
                outputs = model(frames)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * frames.size(0)

        val_loss /= len(val_loader.dataset)
        writer.add_scalar('Loss/val', val_loss, epoch)

        print(f"에포크 [{epoch+1}/{num_epochs}] - 학습 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'checkpoint/best_model.pth')

        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoint/epoch_{epoch+1}.pth')

def validate_model(val_loader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    video_names = []  # 비디오 이름을 저장할 리스트 추가
    with torch.no_grad():
        for frames, labels, video_name in tqdm(val_loader, desc="검증 진행"):
            frames, labels = frames.to(device), labels.to(device).float()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * frames.size(0)
            video_names.extend(video_name)  # 비디오 이름 추가
    val_loss /= len(val_loader.dataset)
    return val_loss, video_names
