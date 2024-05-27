# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm

# # Frame Feature Extractor
# class FrameFeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FrameFeatureExtractor, self).__init__()
#         self.model = timm.create_model('legacy_xception', pretrained=True)
#         self.model.fc = nn.Identity()  # 마지막 FC 레이어 제거
        
#     def forward(self, x):
#         return self.model(x)

# # Temporal Analyzer
# class TemporalAnalyzer(nn.Module):
#     def __init__(self, input_size=512):
#         super(TemporalAnalyzer, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True)
#         self.dropout = nn.Dropout(0.5)
#         self.fc = nn.Linear(128, 128)

#     def forward(self, x):
#         self.lstm.flatten_parameters()
#         lstm_out, _ = self.lstm(x)
#         lstm_out = self.dropout(lstm_out)
#         output = self.fc(lstm_out)
#         return output

# # Spatial-Temporal Encoding
# class SpatialTemporalEncoding(nn.Module):
#     def __init__(self, input_size=2048):
#         super(SpatialTemporalEncoding, self).__init__()
#         self.conv1d_k1 = nn.Conv1d(input_size, 512, kernel_size=1, padding=0)
#         self.conv1d_k2 = nn.Conv1d(input_size, 512, kernel_size=3, padding=1)
#         self.conv1d_k3 = nn.Conv1d(input_size, 512, kernel_size=5, padding=2)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x_k1 = self.relu(self.conv1d_k1(x))
#         x_k2 = self.relu(self.conv1d_k2(x))
#         x_k3 = self.relu(self.conv1d_k3(x))
#         return x_k1, x_k2, x_k3

# # S-MIL Model
# class SMILModel(nn.Module):
#     def __init__(self, frame_feature_extractor, temporal_analyzer):
#         super(SMILModel, self).__init__()
#         self.frame_feature_extractor = frame_feature_extractor
#         self.temporal_analyzer = temporal_analyzer
#         self.spatial_temporal_encoding = SpatialTemporalEncoding(input_size=2048)
#         self.fc1 = nn.Linear(384, 512)  # Changed to 512*3 for combined_encoded
#         self.weight_fc = nn.Linear(512, 1)  # 가중치 계산을 위한 FC 레이어

#     def forward(self, x):
#         batch_size, num_frames, c, h, w = x.size()
#         x = x.view(-1, c, h, w)
#         frame_features = self.frame_feature_extractor(x)
#         frame_features = frame_features.view(batch_size, num_frames, -1)

#         # Spatial-Temporal Encoding
#         x_k1, x_k2, x_k3 = self.spatial_temporal_encoding(frame_features.permute(0, 2, 1))

#         # Max Pooling
#         x_k1 = F.max_pool1d(x_k1, kernel_size=x_k1.size(2)).squeeze(2)
#         x_k2 = F.max_pool1d(x_k2, kernel_size=x_k2.size(2)).squeeze(2)
#         x_k3 = F.max_pool1d(x_k3, kernel_size=x_k3.size(2)).squeeze(2)

#         # 차원 확장 및 결합
#         x_k1 = x_k1.unsqueeze(1).repeat(1, num_frames, 1)
#         x_k2 = x_k2.unsqueeze(1).repeat(1, num_frames, 1)
#         x_k3 = x_k3.unsqueeze(1).repeat(1, num_frames, 1)

#         # 시간적 분석
#         encoded_k1 = self.temporal_analyzer(x_k1)
#         encoded_k2 = self.temporal_analyzer(x_k2)
#         encoded_k3 = self.temporal_analyzer(x_k3)

#         # 인코딩된 특징 결합
#         combined_encoded = torch.cat((encoded_k1, encoded_k2, encoded_k3), dim=2)  # (batch_size, num_frames, 384)

#         # 가중치 계산
#         combined_encoded = combined_encoded.view(batch_size * num_frames, -1)
#         combined_encoded = F.relu(self.fc1(combined_encoded))
#         aij = F.softmax(self.weight_fc(combined_encoded), dim=0).view(batch_size, num_frames, 1)  # (batch_size, num_frames, 1)

#         # Weighted Instances
#         weighted_instances = combined_encoded.view(batch_size, num_frames, -1) * aij

#         # 프레임 수준 예측 계산
#         frame_predictions = torch.sigmoid(weighted_instances.sum(dim=2))  # (batch_size, num_frames)

#         # 비디오 수준 예측 계산 (평균)
#         bag_prediction = frame_predictions.mean(dim=1)  # (batch_size)

#         return bag_prediction

# # S-MIL Loss Function
# def smil_loss(y_pred, y_true):
#     loss = nn.BCELoss()
#     return loss(y_pred, y_true)


import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Frame Feature Extractor
class FrameFeatureExtractor(nn.Module):
    def __init__(self):
        super(FrameFeatureExtractor, self).__init__()
        self.model = timm.create_model('legacy_xception', pretrained=True)
        self.model.fc = nn.Identity()
        
    def forward(self, x):
        return self.model(x)

# Spatial-Temporal Encoding
class SpatialTemporalEncoding(nn.Module):
    def __init__(self, input_size=2048):
        super(SpatialTemporalEncoding, self).__init__()
        self.conv1d_k1 = nn.Conv1d(input_size, 512, kernel_size=1, padding=0)
        self.conv1d_k2 = nn.Conv1d(input_size, 512, kernel_size=3, padding=1)
        self.conv1d_k3 = nn.Conv1d(input_size, 512, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_k1 = self.relu(self.conv1d_k1(x))
        x_k2 = self.relu(self.conv1d_k2(x))
        x_k3 = self.relu(self.conv1d_k3(x))
        return x_k1, x_k2, x_k3

# S-MIL Model
class SMILModel(nn.Module):
    def __init__(self, frame_feature_extractor):
        super(SMILModel, self).__init__()
        self.frame_feature_extractor = frame_feature_extractor
        self.spatial_temporal_encoding = SpatialTemporalEncoding(input_size=2048)
        self.weight_fc = nn.Linear(512 * 3, 1)  # Weighting for instance embeddings

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()  # (8, 250, 3, 255, 255)
        x = x.view(-1, c, h, w)  # Flatten all frames into a single batch (2000, 3, 255, 255)
        frame_features = self.frame_feature_extractor(x)  # Extract frame features
        frame_features = frame_features.view(batch_size, num_frames, -1)  # Reshape to (8, 250, 2048)

        # Spatial-Temporal Encoding
        x_k1, x_k2, x_k3 = self.spatial_temporal_encoding(frame_features.permute(0, 2, 1))  # (8, 2048, 250)

        # Max Pooling
        x_k1 = F.max_pool1d(x_k1, kernel_size=x_k1.size(2)).squeeze(2)  # (8, 512)
        x_k2 = F.max_pool1d(x_k2, kernel_size=x_k2.size(2)).squeeze(2)  # (8, 512)
        x_k3 = F.max_pool1d(x_k3, kernel_size=x_k3.size(2)).squeeze(2)  # (8, 512)

        # Dimension Expansion
        x_k1 = x_k1.unsqueeze(1).repeat(1, num_frames, 1)  # (8, 250, 512)
        x_k2 = x_k2.unsqueeze(1).repeat(1, num_frames, 1)  # (8, 250, 512)
        x_k3 = x_k3.unsqueeze(1).repeat(1, num_frames, 1)  # (8, 250, 512)

        # Combine Encoded Features
        combined_encoded = torch.cat((x_k1, x_k2, x_k3), dim=2)  # (8, 250, 1536)

        # Weight Calculation
        aij = F.softmax(self.weight_fc(combined_encoded), dim=1).view(batch_size, num_frames, 1)  # (8, 250, 1)
        weighted_instances = combined_encoded * aij  # (8, 250, 1536)

        # Frame-Level Predictions
        frame_predictions = torch.sigmoid(weighted_instances.sum(dim=2))  # (8, 250)

        # Bag-Level Predictions (Average)
        bag_prediction = frame_predictions.mean(dim=1)  # (8)

        return bag_prediction

# S-MIL Loss Function
def smil_loss(y_pred, y_true):
    loss = nn.BCELoss()
    return loss(y_pred, y_true)