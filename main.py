import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from dataset import get_dataloader
from model import FrameFeatureExtractor, SMILModel, smil_loss
# from model import TemporalAnalyzer
from train import train_model
from evaluate import evaluate_model, save_results

def main():
    train_base_path = '/home/parkchan/Small_DFDC'
    test_base_path = '/media/NAS/DATASET/1mDFDC/video_level_1mdfdc/validation_set/Test_DFDC'
    batch_size = 8
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 32
    pin_memory = True
    shuffle = True
    validation_split = 0.1
    train_limit = 200

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    print("Creating train and validation dataloaders...")
    train_dataloader, val_dataloader = get_dataloader(train_base_path, batch_size, transform, shuffle, validation_split, num_workers, pin_memory, train_limit)
    print("Dataloaders created.")

    print("Creating test dataloader...")
    test_dataloader, _ = get_dataloader(test_base_path, batch_size, transform, shuffle=False, validation_split=0.0, num_workers=num_workers, pin_memory=pin_memory)
    print("Test dataloader created.")

    frame_feature_extractor = FrameFeatureExtractor()
    # temporal_analyzer = TemporalAnalyzer()
    # model = SMILModel(frame_feature_extractor, temporal_analyzer)
    model = SMILModel(frame_feature_extractor)    
    # GPU 장치 설정
    model = nn.DataParallel(model)
    model.to(device)

    criterion = smil_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    train_model(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs, device)
    print("Training completed.")

    print("Starting evaluation...")
    metrics, confidences, predictions, labels, video_names = evaluate_model(test_dataloader, model, device)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("Evaluation completed.")

    save_results(confidences, predictions, labels, video_names)

if __name__ == "__main__":
    main()