import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
import pandas as pd
import datetime

# 현재 시간을 전역 변수로 설정
now = datetime.datetime.now()
now_str = now.strftime('%Y%m%d_%H%M%S')

def evaluate_model(dataloader, model, device):
    model.eval()
    confidences = []
    predictions = []
    labels = []
    video_names = []  # 비디오 이름을 저장할 리스트 추가
    frame_importances = []  # 각 비디오의 프레임 중요도를 저장할 리스트 추가

    with torch.no_grad():
        for videos, label, video_name in dataloader:
            videos = videos.to(device)
            label = label.to(device)
            
            # 모델 예측
            output, aij = model(videos)
            confidence = output.squeeze().cpu().numpy()
            prediction = (confidence > 0.5).astype(int)

            confidences.append(confidence)
            predictions.append(prediction)
            labels.append(label.cpu().numpy())
            video_names.extend(video_name)  # 비디오 이름 리스트 평탄화
            frame_importances.append(aij.squeeze().cpu().numpy().tolist())
    
    # 리스트의 리스트를 평탄화
    confidences_flat = np.concatenate(confidences)
    predictions_flat = np.concatenate(predictions)
    labels_flat = np.concatenate(labels)

    # 성능 지표 계산
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

def save_results(confidences, predictions, labels, video_names,frame_importances):
    results = pd.DataFrame({
        'video_name': video_names,
        'confidence': confidences,
        'prediction': predictions,
        'label': labels
    })
    results.to_csv(f'{now_str}_results.csv', index=False)

def save_frame_importances(video_names, frame_importances):
    # Create a dictionary to store frame importances for each video
    video_dict = {video: frames for video, frames in zip(video_names, frame_importances)}
    
    # Create a DataFrame from the dictionary
    results = pd.DataFrame.from_dict(video_dict, orient='index')
    
    # Save the DataFrame to a CSV file with the video names as row labels
    results.to_csv(f'{now_str}_frame_importances.csv', index_label='video_name')