import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision import transforms
from dataset import get_dataloader
from model import FrameFeatureExtractor, SMILModel, smil_loss
from train import train_model
from evaluate import evaluate_model, save_results, save_frame_importances
from tqdm import tqdm
from torch.cuda.amp import GradScaler

def main():
    # Initialize the distributed process group
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl')

    print(f"local_rank: {local_rank}")
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    
    
    train_base_path = '/home/parkchan/Small_DFDC'
    test_base_path = '/media/NAS/DATASET/1mDFDC/video_level_1mdfdc/validation_set/Test_DFDC'
    batch_size = 4
    num_epochs = 30
    num_workers = 32
    pin_memory = True
    shuffle = False
    validation_split = 0.2
    train_limit = 100

    transform = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor()
    ])

    print("Creating train and validation dataloaders...")
    train_dataloader, val_dataloader = get_dataloader(train_base_path, batch_size, transform, shuffle, validation_split, num_workers, pin_memory, train_limit, local_rank)
    print("Dataloaders created.")

    print("Creating test dataloader...")
    test_dataloader, _ = get_dataloader(test_base_path, batch_size, transform, shuffle=False, validation_split=0.0, num_workers=num_workers, pin_memory=pin_memory, rank=local_rank)
    print("Test dataloader created.")

    # Test train dataloader step (optional for debugging)
    for videos, labels, _ in tqdm(train_dataloader, desc=f"Test train dataloader step", dynamic_ncols=True):
        videos, labels = videos.to(local_rank), labels.to(local_rank).float()
    print("All ready.")

    weight_root = '/media/NAS/USERS/inho/df_detection/pretrained_model'
    weight_path = os.path.join(weight_root, 'efficient/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23')
    
    # Load weights directly to the correct GPU
    state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage.cuda(local_rank))

    frame_feature_extractor = FrameFeatureExtractor(state_dict)
    model = SMILModel(frame_feature_extractor)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    print(f"Process {dist.get_rank()} using GPU {local_rank}")

    criterion = smil_loss
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    scaler = GradScaler()

    print("Starting training...")
    train_model(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs, local_rank, scaler)
    print("Training completed.")

    print("Starting evaluation...")
    metrics, confidences, predictions, labels, video_names, frame_importances = evaluate_model(test_dataloader, model, local_rank)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("Evaluation completed.")
    save_results(confidences, predictions, labels, video_names, frame_importances)
    save_frame_importances(video_names, frame_importances)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()