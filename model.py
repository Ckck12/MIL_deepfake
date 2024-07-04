# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm

# def replace_nan_to_zero(tensor):
#     if tensor.isnan().any():
#         tensor[tensor.isnan()] = 0.0000000001
#     return tensor

# class FrameFeatureExtractor(nn.Module):
#     def __init__(self, weight_path):
#         super(FrameFeatureExtractor, self).__init__()
#         self.model = timm.create_model('tf_efficientnet_b7_ns', pretrained=False)
        
#         # Load the weights from the provided path
#         checkpoint = torch.load(weight_path)
#         weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

#         # Remove 'encoder.' or 'module.encoder.' prefix if exists
#         weights = {k.replace('encoder.', '').replace('module.', ''): v for k, v in weights.items() if 'epoch' not in k and 'bce_best' not in k}

#         weights.pop("fc.weight", None)
#         weights.pop("fc.bias", None)
        
#         # Load the weights into the model
#         self.model.load_state_dict(weights, strict=True)
        
#         # Remove the final classification layer
#         self.model.classifier = nn.Identity()

#     def forward(self, x):
#         x = self.model(x)
#         x = replace_nan_to_zero(x)  # NaN 처리
#         return x

# class SpatialTemporalEncoding(nn.Module):
#     def __init__(self, input_size=2560):  # EfficientNet-B7 output size
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

# class SMILModel(nn.Module):
#     def __init__(self, frame_feature_extractor):
#         super(SMILModel, self).__init__()
#         self.frame_feature_extractor = frame_feature_extractor
#         self.spatial_temporal_encoding = SpatialTemporalEncoding(input_size=2560)
#         self.fc_alpha = nn.Linear(512 * 3, 1)  # For calculating alpha
#         self.fc_p = nn.Linear(512 * 3, 1)  # For calculating p_i^j
#         nn.init.xavier_uniform_(self.fc_alpha.weight)
#         nn.init.xavier_uniform_(self.fc_p.weight)

#     def forward(self, x):
#         batch_size, num_frames, c, h, w = x.size()  # (batch, num_frames, channels, height, width)

#         frame_features_list = []
#         for i in range(batch_size):
#             frame_features = self.frame_feature_extractor(x[i])  # Extract frame features for each video
#             frame_features_list.append(frame_features)
        
#         frame_features = torch.stack(frame_features_list)  # Combine frame features into a single tensor
#         frame_features = frame_features.view(batch_size, num_frames, -1)  # Reshape to (batch, num_frames, feature_size)
#         frame_features = replace_nan_to_zero(frame_features)
        
#         # Spatial-Temporal Encoding
#         x_k1, x_k2, x_k3 = self.spatial_temporal_encoding(frame_features.permute(0, 2, 1))  # (batch, feature_size, num_frames)

#         # Concatenate encoded features along the feature dimension (i.e., dim=1)
#         combined_encoded = torch.cat((x_k1, x_k2, x_k3), dim=1)  # (batch, 512*3, num_frames)
#         combined_encoded = replace_nan_to_zero(combined_encoded)

#         # Calculate alpha_ij (weights for instances)
#         alpha_logits = self.fc_alpha(combined_encoded.permute(0, 2, 1))  # (batch, num_frames, 1)
#         aij = F.softmax(alpha_logits, dim=1).permute(0, 2, 1)  # (batch, 1, num_frames)

#         # Calculate fake scores for each frame
#         p_i_j_logits = self.fc_p(combined_encoded.permute(0, 2, 1))  # (batch, num_frames, 1)
#         p_i_j = torch.sigmoid(p_i_j_logits).squeeze(2)  # (batch, num_frames)

#         # Calculate final bag prediction using proposed method
#         weighted_p_i_j = torch.prod((1 / p_i_j - 1).pow(aij.squeeze()), dim=1)

#         # bag prediction이 0과 1사이가 아니라면 이번 데이터는 학습시키지 말고 다음 데이터로 넘어가도록 하기 
#         bag_prediction = 1 / (1 + weighted_p_i_j)  # (batch)
        
#         if bag_prediction.isnan().any() or ((bag_prediction < 0) | (bag_prediction > 1)).any():
#             raise ValueError("Invalid bag_prediction values")
        
#         return bag_prediction, aij

# # S-MIL Loss Function
# def smil_loss(y_pred, y_true):
#     loss = nn.BCELoss()
#     return loss(y_pred, y_true)
# --------------------------------------------------------------------------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm

# def replace_nan_to_zero(tensor):
#     if tensor.isnan().any():
#         tensor[tensor.isnan()] = 0.0000000001
#     return tensor

# class FrameFeatureExtractor(nn.Module):
#     def __init__(self, weight_path):
#         super(FrameFeatureExtractor, self).__init__()
#         self.model = timm.create_model('tf_efficientnet_b7_ns', pretrained=False)
        
#         # Load the weights from the provided path
#         checkpoint = torch.load(weight_path)
#         weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

#         # Remove 'encoder.' or 'module.encoder.' prefix if exists
#         weights = {k.replace('encoder.', '').replace('module.', ''): v for k, v in weights.items() if 'epoch' not in k and 'bce_best' not in k}

#         weights.pop("fc.weight", None)
#         weights.pop("fc.bias", None)
        
#         # Load the weights into the model
#         self.model.load_state_dict(weights, strict=True)
        
#         # Remove the final classification layer
#         self.model.classifier = nn.Identity()

#     def forward(self, x):
#         x = self.model(x)
#         x = replace_nan_to_zero(x)
#         return x

# class SpatialTemporalEncoding(nn.Module):
#     def __init__(self, input_size=2560):  # EfficientNet-B7 output size
#         super(SpatialTemporalEncoding, self).__init__()
#         self.conv1d_k1 = nn.Conv1d(input_size, 512, kernel_size=1, padding=0)
#         self.conv1d_k2 = nn.Conv1d(input_size, 512, kernel_size=2, padding=1)
#         self.conv1d_k3 = nn.Conv1d(input_size, 512, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x_k1 = self.relu(self.conv1d_k1(x))
#         x_k2 = self.relu(self.conv1d_k2(x))
#         x_k3 = self.relu(self.conv1d_k3(x))
#         return x_k1, x_k2, x_k3

# class SMILModel(nn.Module):
#     def __init__(self, frame_feature_extractor):
#         super(SMILModel, self).__init__()
#         self.frame_feature_extractor = frame_feature_extractor
#         self.spatial_temporal_encoding = SpatialTemporalEncoding(input_size=2560)
#         self.fc_alpha = nn.Linear(512, 1)
#         self.fc_p = nn.Linear(512, 1)
#         nn.init.xavier_uniform_(self.fc_alpha.weight)
#         nn.init.xavier_uniform_(self.fc_p.weight)

#     def forward(self, x):
#         batch_size, num_frames, c, h, w = x.size()

#         frame_features_list = []
#         for i in range(batch_size):
#             frame_features = self.frame_feature_extractor(x[i])
#             frame_features_list.append(frame_features)
        
#         frame_features = torch.stack(frame_features_list)
#         frame_features = frame_features.view(batch_size, num_frames, -1)
#         frame_features = replace_nan_to_zero(frame_features)
        
#         # Spatial-Temporal Encoding
#         x_k1, x_k2, x_k3 = self.spatial_temporal_encoding(frame_features.permute(0, 2, 1))

#         bag_predictions = []
#         for x_k in [x_k1, x_k2, x_k3]:
#             combined_encoded = replace_nan_to_zero(x_k)
#             alpha_logits = self.fc_alpha(combined_encoded.permute(0, 2, 1))
#             aij = F.softmax(alpha_logits, dim=1).permute(0, 2, 1)

#             p_i_j_logits = self.fc_p(combined_encoded.permute(0, 2, 1))
#             p_i_j = torch.sigmoid(p_i_j_logits).squeeze(2)

#             weighted_p_i_j = torch.prod((1 / p_i_j - 1).pow(aij.squeeze()), dim=1)
#             bag_prediction = 1 / (1 + weighted_p_i_j)
#             bag_predictions.append(bag_prediction)
        
#         # Compute the final prediction using the weighted product aggregation
#         final_bag_prediction = torch.prod(torch.stack(bag_predictions), dim=0)

#         if final_bag_prediction.isnan().any() or ((final_bag_prediction < 0) | (final_bag_prediction > 1)).any():
#             raise ValueError("Invalid bag_prediction values")
        
#         return final_bag_prediction, aij

# # S-MIL Loss Function
# def smil_loss(y_pred, y_true):
#     loss = nn.BCELoss()
#     return loss(y_pred, y_true)


#--------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def replace_nan_to_zero(tensor):
    if tensor.isnan().any():
        tensor[tensor.isnan()] = 0.0000000001
    return tensor

class FrameFeatureExtractor(nn.Module):
    def __init__(self, state_dict):
        super(FrameFeatureExtractor, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b7_ns', pretrained=False)
        
        # Load the weights from the provided state_dict
        weights = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict

        # Remove 'encoder.' or 'module.encoder.' prefix if exists
        weights = {k.replace('encoder.', '').replace('module.', ''): v for k, v in weights.items() if 'epoch' not in k and 'bce_best' not in k}

        weights.pop("fc.weight", None)
        weights.pop("fc.bias", None)
        
        # Load the weights into the model
        self.model.load_state_dict(weights, strict=True)
        
        # Remove the final classification layer
        self.model.classifier = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        x = replace_nan_to_zero(x)
        return x

class SpatialTemporalEncoding(nn.Module):
    def __init__(self, input_size=2560):  # EfficientNet-B7 output size
        super(SpatialTemporalEncoding, self).__init__()
        self.conv1d_k1 = nn.Conv1d(input_size, 512, kernel_size=1, padding=0)
        self.conv1d_k2 = nn.Conv1d(input_size, 512, kernel_size=2, padding=1)
        self.conv1d_k3 = nn.Conv1d(input_size, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_k1 = self.relu(self.conv1d_k1(x))
        x_k2 = self.relu(self.conv1d_k2(x))
        x_k3 = self.relu(self.conv1d_k3(x))
        return x_k1, x_k2, x_k3

class SMILModel(nn.Module):
    def __init__(self, frame_feature_extractor):
        super(SMILModel, self).__init__()
        self.frame_feature_extractor = frame_feature_extractor
        self.spatial_temporal_encoding = SpatialTemporalEncoding(input_size=2560)
        self.fc_alpha = nn.Linear(512, 1)
        self.fc_w = nn.Linear(512, 1)
        nn.init.xavier_uniform_(self.fc_alpha.weight)
        nn.init.xavier_uniform_(self.fc_w.weight)

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()

        frame_features_list = []
        for i in range(batch_size):
            frame_features = self.frame_feature_extractor(x[i])
            frame_features_list.append(frame_features)
        
        frame_features = torch.stack(frame_features_list)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        frame_features = replace_nan_to_zero(frame_features)
        
        # Spatial-Temporal Encoding
        x_k1, x_k2, x_k3 = self.spatial_temporal_encoding(frame_features.permute(0, 2, 1))

        all_aij = []
        all_pij = []

        for x_k in [x_k1, x_k2, x_k3]:
            combined_encoded = replace_nan_to_zero(x_k)
            
            # Calculate alpha_i^j
            alpha_logits = self.fc_alpha(combined_encoded.permute(0, 2, 1))  # (batch_size, num_frames, 1)
            aij = F.softmax(alpha_logits, dim=1).permute(0, 2, 1)  # (batch_size, 1, num_frames)
            all_aij.append(aij.squeeze(1))  # (batch_size, num_frames)

            # Calculate p_i^j
            w_logits = self.fc_w(combined_encoded.permute(0, 2, 1))  # (batch_size, num_frames, 1)
            p_i_j = torch.sigmoid(w_logits).squeeze(2)  # (batch_size, num_frames)
            all_pij.append(p_i_j)
            
        # Find the maximum num_frames across all batches
        max_num_frames = max([tensor.size(1) for tensor in all_aij])

        # Pad all tensors to the same size
        all_aij = [F.pad(tensor, (0, max_num_frames - tensor.size(1)), 'constant', 0) for tensor in all_aij]
        all_pij = [F.pad(tensor, (0, max_num_frames - tensor.size(1)), 'constant', 0) for tensor in all_pij]
        
        all_aij = torch.stack(all_aij)  # (3, batch_size, num_frames)
        all_pij = torch.stack(all_pij)  # (3, batch_size, num_frames)

        # Final bag prediction using the provided formula
        weighted_p_i_j = torch.prod((1 / all_pij - 1).pow(all_aij), dim=2)  # (3, batch_size)
        bag_predictions = 1 / (1 + weighted_p_i_j)  # (3, batch_size)
        print(f"bag_predictions: {bag_predictions}")
        final_bag_prediction = torch.prod(bag_predictions, dim=0)  # (batch_size)
        print(f"final_bag_prediction: {final_bag_prediction}")
        if final_bag_prediction.isnan().any() or ((final_bag_prediction < 0) | (final_bag_prediction > 1)).any():
            raise ValueError("Invalid bag_prediction values")
        
        return final_bag_prediction, all_aij

# S-MIL Loss Function
def smil_loss(y_pred, y_true, weight):
    pos_weight = torch.tensor([weight], device=y_pred.device)
    return F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=pos_weight)
