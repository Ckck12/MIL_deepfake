import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from dataset import get_dataloader
from model_shahid import FrameFeatureExtractor, SMILModel, smil_loss
from train_shahid import train_model
from evaluate import evaluate_model, save_results, save_frame_importances
from tqdm import tqdm
def main():
    train_base_path = '/media/NAS/DATASET/1mDFDC/video_level_1mdfdc/train_set/DFDC'
    test_base_path = '/media/NAS/DATASET/1mDFDC/video_level_1mdfdc/validation_set/Test_DFDC'
    batch_size = 4
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 32
    pin_memory = True
    shuffle = False
    validation_split = 0.1
    train_limit = 100000

    transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.ToTensor()
    ])

    print("Creating train and validation dataloaders...")
    train_dataloader, val_dataloader = get_dataloader(train_base_path, batch_size, transform, shuffle, validation_split, num_workers, pin_memory, train_limit)
    print("Dataloaders created.")

    print("Creating test dataloader...")
    test_dataloader, _ = get_dataloader(test_base_path, batch_size, transform, shuffle=False, validation_split=0.0, num_workers=num_workers, pin_memory=pin_memory)
    print("Test dataloader created.")
    for videos, labels, _ in tqdm(train_dataloader, desc=f"Test train dataloader step", dynamic_ncols=True):
            videos, labels = videos.to(device), labels.to(device).float()
    print("All ready.")
    # 가중치 경로 설정
    weight_root = '/media/NAS/USERS/inho/df_detection/pretrained_model'
    weight_path = os.path.join(weight_root, 'efficient/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23')

    # FrameFeatureExtractor 인스턴스 생성 시 가중치 경로 전달
    frame_feature_extractor = FrameFeatureExtractor(weight_path)

    model = SMILModel(frame_feature_extractor)    
    model = nn.DataParallel(model)
    model.to(device)

    criterion = smil_loss
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    print("Starting training...")
    train_model(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs, device)
    print("Training completed.")

    print("Starting evaluation...")
    metrics, confidences, predictions, labels, video_names, frame_importances = evaluate_model(test_dataloader, model, device)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("Evaluation completed.")
    save_results(confidences, predictions, labels, video_names, frame_importances)
    save_frame_importances(video_names, frame_importances)
if __name__ == "__main__":
    main()
