import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
import pandas as pd

def evaluate_model(dataloader, model, device):
    model.eval()
    confidences = []
    predictions = []
    labels = []
    video_names = []  # 비디오 이름을 저장할 리스트 추가

    with torch.no_grad():
        for frames, label, video_name in dataloader:
            frames = frames.to(device)
            outputs = model(frames)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            print(f"Video: {video_name}, Outputs: {outputs}, Probs: {probs}, Preds: {preds}")

            confidences.extend(probs.cpu().numpy().flatten().tolist())
            predictions.extend(preds.cpu().numpy().flatten().tolist())
            labels.extend(label.cpu().numpy().flatten().tolist())
            video_names.extend(video_name)  # 비디오 이름 추가

    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'roc_auc': roc_auc_score(labels, confidences),
        'f1': f1_score(labels, predictions),
        'recall': recall_score(labels, predictions),
        'precision': precision_score(labels, predictions)
    }

    return metrics, confidences, predictions, labels, video_names

def save_results(confidences, predictions, labels, video_names):
    results = pd.DataFrame({
        'video_name': video_names,
        'confidence': confidences,
        'prediction': predictions,
        'label': labels
    })
    results.to_csv('results.csv', index=False)