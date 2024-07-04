import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def replace_nan_to_zero(tensor):
    if tensor.isnan().any():
        tensor[tensor.isnan()] = 0.0000000001
    return tensor

class FrameFeatureExtractor(nn.Module):
    def __init__(self, weight_path):
        super(FrameFeatureExtractor, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b7_ns', pretrained=False)
        
        # Load the weights from the provided path
        checkpoint = torch.load(weight_path)
        weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

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
        x = replace_nan_to_zero(x)  # NaN 처리
        return x

class SpatialTemporalEncoding(nn.Module):
    def __init__(self, input_size=2560):  # EfficientNet-B7 output size
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

class SMILModel(nn.Module):
    def __init__(self, frame_feature_extractor):
        super(SMILModel, self).__init__()
        self.frame_feature_extractor = frame_feature_extractor
        self.spatial_temporal_encoding = SpatialTemporalEncoding(input_size=2560)
        self.fc_alpha = nn.Linear(1536 * 48, 1)  # For calculating alpha
        # self.fc_alpha2 = nn.Linear(512, 1)  # For calculating alpha
        self.fc_p = nn.Linear(1536 * 48, 1)  # For calculating p_i^j
        # self.fc_p2 = nn.Linear(512, 1)  # For calculating alpha
        nn.init.xavier_uniform_(self.fc_alpha.weight)
        # nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_p.weight)
        # nn.init.xavier_uniform_(self.fc_p2.weight)

    def forward(self, x):
        # batch_size, num_frames, c, h, w = x.size()  # (batch, num_frames, channels, height, width)
        #x = x.view(-1, c, h, w)  # Flatten all frames into a single batch (batch, channels*num_frames, height, width)/
        # x = torch.flatten(x, start_dim=1, end_dim=2)
        features=[]
        for image in x:
            features.append(self.frame_feature_extractor(image))  # Extract frame features
        frame_features = torch.stack(features)

        # Spatial-Temporal Encoding
        x_k1, x_k2, x_k3 = self.spatial_temporal_encoding(frame_features.permute(0, 2, 1))  # (batch, feature_size, num_frames)

        # Concatenate encoded features along the feature dimension (i.e., dim=1)
        combined_encoded = torch.cat((x_k1, x_k2, x_k3), dim=1)  # (batch, 512*3, num_frames)
        

        # Calculate alpha_ij (weights for instances)
        combined_encoded = torch.flatten(combined_encoded, start_dim=1, end_dim=2)  # (batch, 512*3*num_frames
        alpha_logits = self.fc_alpha(combined_encoded)  # (batch, num_frames, 512)
        # alpha_logits = self.fc_alpha2(alpha_logits)  # (512, 1)
        aij = F.softmax(alpha_logits, dim=1)  # (batch, 1)

        # Calculate fake scores for each frame
        p_i_j_logits = self.fc_p(combined_encoded)  # (batch, 512)
        # p_i_j_logits = self.fc_p2(p_i_j_logits)  # (512, 1)
        p_i_j = torch.sigmoid(p_i_j_logits)  # (batch, 1)

        # Calculate final bag prediction using proposed method
        weighted_p_i_j = torch.prod(((1 / p_i_j) - 1).pow(aij.squeeze()), dim=1)
        bag_prediction = 1 / (1 + weighted_p_i_j)  # (batch)/
        # print(f"bag_prediction: {bag_prediction}")
        
        # print(f"'0<'{(bag_prediction >= 0) & (bag_prediction <= 1)}'<1'")
        # print("---------------------------------------------------")

        if bag_prediction.isnan().any() or ((bag_prediction < 0) | (bag_prediction > 1)).any():
            raise ValueError("Invalid bag_prediction values")
        
        return bag_prediction, aij

# S-MIL Loss Function
def smil_loss(y_pred, y_true):
    loss = nn.BCELoss()
    return loss(y_pred, y_true)
