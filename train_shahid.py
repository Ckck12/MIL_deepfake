import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
from dataset import get_dataloader
from model_shahid import FrameFeatureExtractor, SMILModel, smil_loss

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, device):
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    
    best_val_loss = float('inf')
    writer = SummaryWriter()  # For TensorBoard logging

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total = 0
        correct = 0
        all_labels = []
        all_predictions = []
        
        # Training loop
        for videos, labels, _ in tqdm(train_loader, desc=f"에포크 [{epoch+1}/{num_epochs}] 학습 진행", dynamic_ncols=True):
            videos, labels = videos.to(device), labels.to(device).float()
            optimizer.zero_grad()
            try:
                outputs, _ = model(videos) # BF x 1 -> B x F x 1
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() 
                
                predicted = (outputs > 0.5).float()

                # Ensure the shapes of predicted and labels are compatible
                predicted = predicted.view_as(labels)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect labels and predictions for final accuracy calculation
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
            except ValueError as e:
                print(f"Skipping batch due to error: {e}")
                continue

        train_loss /= len(train_loader.dataset)
        writer.add_scalar('Loss/train', train_loss, epoch)
        train_acc = correct / total

        # Calculate final training accuracy using accuracy_score
        final_train_accuracy = accuracy_score(all_labels, all_predictions)
        
        val_loss, val_acc, final_val_accuracy, _ = validate_model(val_loader, model, criterion, device)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"에포크 [{epoch+1}/{num_epochs}] - train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, final train acc: {final_train_accuracy:.4f} val loss: {val_loss:.4f}, val acc: {val_acc:.4f}, final val acc: {final_val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            now = datetime.datetime.now()
            # Format as string
            now_str = now.strftime('%Y%m%d_%H%M%S')
            # Save the best model name with date and time
            torch.save(model.state_dict(), f'checkpoint/best_model_{now_str}.pth')
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'checkpoint/epoch_{epoch+1}_{now_str}.pth')

    writer.close()

def validate_model(val_loader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    video_names = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for videos, labels, video_name in tqdm(val_loader, desc="검증 진행", dynamic_ncols=True):
            videos, labels = videos.to(device), labels.to(device).float()
            try:
                outputs, _ = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item() 
                
                predicted = (outputs > 0.5).float()

                # Ensure the shapes of predicted and labels are compatible
                predicted = predicted.view_as(labels)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                video_names.extend(video_name)
                
                # Collect labels and predictions for final accuracy calculation
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
            except ValueError as e:
                print(f"Skipping batch due to error: {e}")
                continue

    val_loss /= len(val_loader.dataset)
    val_acc = correct / total

    # Calculate final validation accuracy using accuracy_score
    final_val_accuracy = accuracy_score(all_labels, all_predictions)
    
    return val_loss, val_acc, final_val_accuracy, video_names

# Add your model initialization and training call here
