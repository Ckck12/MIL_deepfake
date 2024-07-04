import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from torch.utils.data.distributed import DistributedSampler

class DeepFakeDataset(Dataset):
    def __init__(self, base_path, transform=None, frame_limit=52, train_limit=None):
        self.base_path = base_path
        self.transform = transform
        self.frame_limit = frame_limit
        self.data = []
        self.video_names = []  # 비디오 이름을 저장할 리스트 추가

        video_folders = os.listdir(base_path)
        if train_limit:
            video_folders = video_folders[:train_limit]  # train_limit을 적용하여 비디오 폴더 수를 제한

        for video_folder in video_folders:
            folder_path = os.path.join(base_path, video_folder)
            frames = sorted(glob.glob(os.path.join(folder_path, '*.png')))
            if frames:
                selected_frames = self.select_frames(frames)
                if len(selected_frames) == 0:
                    print(f"Warning: No frames left after selection in folder {video_folder}")
                    continue  # 프레임이 0개인 경우 이 비디오 로드하지 않음
                label = 1 if 'fake' in video_folder.lower() else 0
                self.data.append((selected_frames, label))
                self.video_names.append(video_folder)
            else:
                print(f"Warning: No frames found in folder {video_folder}")

        print(f"총 {len(self.data)}개의 비디오가 로드되었습니다.")

    def select_frames(self, frames):
        # 한프레임 가져오고 두프레임 건너뛰기
        selected_frames = frames[::3]
        return selected_frames[:self.frame_limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames, label = self.data[idx]
        video_name = self.video_names[idx]
        images = []
        
        for frame in frames:
            try:
                image = Image.open(frame).convert('RGB')
                if self.transform:
                    image = self.transform(image)  # 3 x H x W
                images.append(image)
            except (UnidentifiedImageError, OSError) as e:
                print(f"Warning: Unable to load frame {frame}: {e}")
                continue
        if not images:
            raise ValueError(f"Error: No valid images found for index {idx} - {video_name} - {frames}")

        # 프레임이 부족한 경우 패딩 추가
        if len(images) < self.frame_limit:
            padding_count = self.frame_limit - len(images)
            padding = [torch.zeros_like(images[0]) for _ in range(padding_count)]
            images.extend(padding)

        images = torch.stack(images)  # 3F x H x W

        # images에 nan이 포함되어 있는지 확인
        if images.isnan().any():
            raise ValueError(f"Error: NaN found in images for index {idx} - {video_name}")

        label = torch.tensor(label, dtype=torch.float32)
        return images, label, video_name

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    max_length = max([sample[0].shape[0] for sample in batch])
    for i, (images, label, video_name) in enumerate(batch):
        if images.shape[0] < max_length:
            padding = torch.zeros((max_length - images.shape[0], *images.shape[1:]))
            images = torch.cat((images, padding), dim=0)
        batch[i] = (images, label, video_name)
    return default_collate(batch)

def get_dataloader(base_path, batch_size=8, transform=None, shuffle=True, validation_split=0.2, num_workers=8, pin_memory=True, train_limit=None, rank=None):
    dataset = DeepFakeDataset(base_path, transform=transform, train_limit=train_limit)
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DistributedSamplers
    world_size = torch.distributed.get_world_size()  # 전체 프로세스 수
    print(f"world_size: {world_size}")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=pin_memory, collate_fn=custom_collate_fn)

    return train_dataloader, val_dataloader