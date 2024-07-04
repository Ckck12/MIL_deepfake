import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
import pandas as pd
import datetime
import torch.distributed as dist

# 현재 시간을 전역 변수로 설정
now = datetime.datetime.now()
now_str = now.strftime('%Y%m%d_%H%M%S')

def gather_from_all_processes(data, world_size):
    gathered_data = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_data, data)
    return [item for sublist in gathered_data for item in sublist]

def evaluate_model(dataloader, model, device):
    model.eval()
    confidences = []
    predictions = []
    labels = []
    video_names = []
    frame_importances = []

    with torch.no_grad():
        for videos, label, video_name in dataloader:
            videos = videos.to(device)
            label = label.to(device)
            
            with torch.cuda.amp.autocast():  # Mixed precision inference
                output, aij = model(videos)
                
            confidence = output.squeeze().cpu().numpy()
            prediction = (confidence > 0.5).astype(int)

            confidences.append(confidence)
            predictions.append(prediction)
            labels.append(label.cpu().numpy())
            video_names.extend(video_name)
            frame_importances.append(aij.squeeze().cpu().numpy().tolist())
    
    # Gather results from all processes
    world_size = dist.get_world_size()
    confidences = gather_from_all_processes(confidences, world_size)
    predictions = gather_from_all_processes(predictions, world_size)
    labels = gather_from_all_processes(labels, world_size)
    video_names = gather_from_all_processes(video_names, world_size)
    frame_importances = gather_from_all_processes(frame_importances, world_size)

    # Flatten lists
    confidences_flat = np.concatenate(confidences)
    predictions_flat = np.concatenate(predictions)
    labels_flat = np.concatenate(labels)

    # Calculate metrics
    acc = accuracy_score(labels_flat, predictions_flat)
    roc_auc = roc_auc_score(labels_flat, confidences_flat)
    f1 = f1_score(labels_flat, predictions_flat)
    recall = recall_score(labels_flat, predictions_flat)
    precision = precision_score(labels_flat, predictions_flat)

    print(f'Accuracy: {acc:.4f}')
    print(f'ROC AUC Score: {roc_auc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')

    metrics = {
        'Accuracy': acc,
        'ROC AUC Score': roc_auc,
        'F1 Score': f1,
        'Recall': recall,
        'Precision': precision
    }

    return metrics, confidences_flat.tolist(), predictions_flat.tolist(), labels_flat.tolist(), video_names, frame_importances

def save_results(confidences, predictions, labels, video_names, frame_importances):
    if dist.get_rank() == 0:
        results = pd.DataFrame({
            'video_name': video_names,
            'confidence': confidences,
            'prediction': predictions,
            'label': labels
        })
        results.to_csv(f'{now_str}_results.csv', index=False)

def save_frame_importances(video_names, frame_importances):
    if dist.get_rank() == 0:
        # Create a dictionary to store frame importances for each video
        video_dict = {video: frames for video, frames in zip(video_names, frame_importances)}
        
        # Create a DataFrame from the dictionary
        results = pd.DataFrame.from_dict(video_dict, orient='index')
        
        # Save the DataFrame to a CSV file with the video names as row labels
        results.to_csv(f'{now_str}_frame_importances.csv', index_label='video_name')
