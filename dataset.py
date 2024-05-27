import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from PIL import Image

class DeepFakeDataset(Dataset):
    def __init__(self, base_path, transform=None, frame_limit=50):
        self.base_path = base_path
        self.transform = transform
        self.frame_limit = frame_limit
        self.data = []
        self.video_names = []  # 비디오 이름을 저장할 리스트 추가
        for video_folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, video_folder)
            frames = sorted(glob.glob(os.path.join(folder_path, '*.png')))  # 이미지 파일 형식을 png로 수정
            if frames:  # 프레임 리스트가 비어 있지 않은 경우에만 데이터 추가
                if len(frames) > self.frame_limit:
                    frames = frames[:self.frame_limit]  # 프레임 수를 제한
                label = 'fake' in video_folder.lower()  # 'fake'이 폴더 이름에 포함되어 있으면 label을 1로 설정
                self.data.append((frames, label))
                self.video_names.append(video_folder)  # 비디오 이름 추가
            else:
                print(f"Warning: No frames found in folder {video_folder}")
        print(f"총 {len(self.data)}개의 비디오가 로드되었습니다.")  # 데이터셋의 크기 출력

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames, label = self.data[idx]
        video_name = self.video_names[idx]  # 비디오 이름 가져오기
        images = []
        for frame in frames:
            image = Image.open(frame).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        if not images:  # images 리스트가 비어 있으면 예외 처리
            print(f"Error: No images found for index {idx}")
            raise ValueError(f"인덱스 {idx}에 대한 이미지가 없습니다.")
        images = torch.stack(images)
        label = torch.tensor(label, dtype=torch.float32)
        return images, label, video_name

def custom_collate_fn(batch):
    # 배치 내의 샘플들을 동일한 크기로 맞춤
    max_length = max([sample[0].shape[0] for sample in batch])
    for i, (images, label, video_name) in enumerate(batch):
        if images.shape[0] < max_length:
            padding = torch.zeros((max_length - images.shape[0], *images.shape[1:]))
            images = torch.cat((images, padding), dim=0)
        batch[i] = (images, label, video_name)
    return default_collate(batch)

def get_dataloader(base_path, batch_size=8, transform=None, shuffle=True, validation_split=0.2, num_workers=4, pin_memory=True, train_limit=None):
    dataset = DeepFakeDataset(base_path, transform=transform)

    # 데이터셋의 일부분만 학습에 사용하도록 제한
    if train_limit:
        dataset = torch.utils.data.Subset(dataset, range(train_limit))

    # 데이터셋을 학습 및 검증 데이터셋으로 분할
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=custom_collate_fn)

    return train_dataloader, val_dataloader
