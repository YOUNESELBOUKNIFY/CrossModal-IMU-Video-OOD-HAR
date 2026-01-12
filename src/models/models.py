"""
Modèles pour IMU-Video Cross-modal Learning
Inspirés de l'article (PatchTST pour IMU, VideoMAE pour vidéo)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models


# ============ IMU Encoder (PatchTST-inspired) ============

class PatchEmbedding(nn.Module):
    """Embedding par patches pour séries temporelles"""
    
    def __init__(self, in_channels, patch_size, stride, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        
        # Projection linéaire par canal
        self.projections = nn.ModuleList([
            nn.Linear(patch_size, d_model) for _ in range(in_channels)
        ])
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, channels, num_patches, d_model)
        """
        B, C, L = x.shape
        
        # Unfold pour créer des patches
        patches = x.unfold(dimension=2, size=self.patch_size, step=self.stride)
        # patches: (batch, channels, num_patches, patch_size)
        
        # Projeter chaque canal indépendamment
        embedded = []
        for c in range(C):
            proj = self.projections[c](patches[:, c])  # (B, num_patches, d_model)
            embedded.append(proj)
        
        embedded = torch.stack(embedded, dim=1)  # (B, C, num_patches, d_model)
        return embedded


class IMUEncoder(nn.Module):
    """
    Encodeur IMU basé sur PatchTST
    Traite chaque canal IMU comme une série univariée indépendante
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config.model
        
        self.in_channels = config.data.imu_channels
        self.patch_size = model_cfg.imu_patch_size
        self.stride = model_cfg.imu_stride
        self.d_model = model_cfg.imu_d_model
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            self.in_channels,
            self.patch_size,
            self.stride,
            self.d_model
        )
        
        # CLS token pour représentation globale
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        
        # Positional encoding
        max_patches = (config.data.imu_window_size - self.patch_size) // self.stride + 1
        self.pos_encoding = nn.Parameter(torch.randn(1, max_patches + 1, self.d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=model_cfg.imu_nhead,
            dim_feedforward=self.d_model * 4,
            dropout=model_cfg.imu_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_cfg.imu_num_layers
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(self.d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            embeddings: (batch, d_model) - représentation du CLS token
            all_tokens: (batch, num_patches+1, d_model) - tous les tokens
        """
        B = x.shape[0]
        
        # Patch embedding: (B, C, num_patches, d_model)
        patches = self.patch_embed(x)
        
        # Flatten channels et patches: (B, C*num_patches, d_model)
        B, C, N, D = patches.shape
        patches = patches.reshape(B, C * N, D)
        
        # Ajouter CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)  # (B, 1+C*N, d_model)
        
        # Positional encoding (truncate si nécessaire)
        pos_len = min(tokens.shape[1], self.pos_encoding.shape[1])
        tokens = tokens[:, :pos_len] + self.pos_encoding[:, :pos_len]
        
        # Transformer
        encoded = self.transformer(tokens)
        encoded = self.norm(encoded)
        
        # CLS token comme représentation globale
        cls_output = encoded[:, 0]  # (B, d_model)
        
        return cls_output, encoded


# ============ Video Encoder (Lightweight) ============

class VideoEncoder(nn.Module):
    """
    Encodeur vidéo simplifié basé sur CNN 2D + pooling temporel
    Pour version production, utiliser VideoMAE ou autre
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config.model
        
        # Backbone CNN (ResNet18 ou MobileNet)
        if model_cfg.video_backbone == 'resnet18':
            backbone = models.resnet18(pretrained=model_cfg.video_pretrained)
            self.feature_dim = 512
        elif model_cfg.video_backbone == 'mobilenet_v2':
            backbone = models.mobilenet_v2(pretrained=model_cfg.video_pretrained)
            self.feature_dim = 1280
        else:
            raise ValueError(f"Backbone inconnu: {model_cfg.video_backbone}")
        
        # Enlever la dernière couche FC
        if model_cfg.video_backbone == 'MCG-NJU/videomae-base-ssv2':
            self.is_videomae = True
            self.backbone = VideoMAEModel.from_pretrained(
                model_cfg.video_backbone
            )
            self.feature_dim = self.backbone.config.hidden_size  # 768

        else:
            raise ValueError(f"Backbone inconnu: {model_cfg.video_backbone}")

        # ======================
        # 🔹 Projection
        # ======================
        self.projection = nn.Linear(
            self.feature_dim,
            model_cfg.video_d_model
        )

        # CNN seulement
        if not self.is_videomae:
            self.temporal_pool = nn.AdaptiveAvgPool1d(1)
            
    def forward(self, x):
        """
        Args:
            x: (batch, num_frames, C, H, W)
        Returns:
            embeddings: (batch, video_d_model)
        """
        B, T, C, H, W = x.shape
        
        # Reshape pour traiter toutes les frames ensemble
        x = x.view(B * T, C, H, W)
        
        # Extract features
        features = self.backbone(x)  # (B*T, feature_dim, h, w)
        
        # Global average pooling spatial
        features = F.adaptive_avg_pool2d(features, (1, 1))  # (B*T, feature_dim, 1, 1)
        features = features.view(B, T, self.feature_dim)  # (B, T, feature_dim)
        
        # Projection
        features = self.projection(features)  # (B, T, video_d_model)
        
        # Temporal pooling
        features = features.transpose(1, 2)  # (B, video_d_model, T)
        pooled = self.temporal_pool(features).squeeze(-1)  # (B, video_d_model)
        
        return pooled


# ============ Projection Heads ============

class ProjectionHead(nn.Module):
    """Projection head pour contrastive learning"""
    
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)


# ============ Cross-Modal Model ============

class CrossModalModel(nn.Module):
    """
    Modèle complet pour pretraining cross-modal IMU-Video
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config.model
        
        # Encoders
        self.imu_encoder = IMUEncoder(config)
        self.video_encoder = VideoEncoder(config)
        
        # Projection heads
        self.imu_proj = ProjectionHead(
            model_cfg.imu_d_model,
            model_cfg.projection_hidden_dim,
            model_cfg.projection_dim
        )
        
        self.video_proj = ProjectionHead(
            model_cfg.video_d_model,
            model_cfg.projection_hidden_dim,
            model_cfg.projection_dim
        )
        
        # Temperature et bias pour sigmoid loss
        self.temperature = nn.Parameter(torch.ones([]) * math.log(10))
        self.bias = nn.Parameter(torch.ones([]) * -10)
    
    def forward(self, imu, video):
        """
        Args:
            imu: (batch, channels, seq_len)
            video: (batch, num_frames, C, H, W)
        Returns:
            imu_proj: (batch, projection_dim)
            video_proj: (batch, projection_dim)
        """
        # Encode
        imu_feat, _ = self.imu_encoder(imu)
        video_feat = self.video_encoder(video)
        
        # Project
        imu_proj = self.imu_proj(imu_feat)
        video_proj = self.video_proj(video_feat)
        
        # L2 normalize
        imu_proj = F.normalize(imu_proj, dim=1)
        video_proj = F.normalize(video_proj, dim=1)
        
        return imu_proj, video_proj


# ============ Classification Model ============

class IMUClassifier(nn.Module):
    """
    Classificateur pour activités basé sur IMU encoder pré-entraîné
    """
    
    def __init__(self, imu_encoder, config, freeze_encoder=False):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.config = config
        model_cfg = config.model
        
        if freeze_encoder:
            for param in self.imu_encoder.parameters():
                param.requires_grad = False
        
        # Classification head
        layers = []
        in_dim = model_cfg.imu_d_model
        
        for hidden_dim in model_cfg.classifier_hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(model_cfg.classifier_dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, model_cfg.num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, imu):
        """
        Args:
            imu: (batch, channels, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        with torch.set_grad_enabled(self.training or not self.freeze_encoder):
            imu_feat, _ = self.imu_encoder(imu)
        
        logits = self.classifier(imu_feat)
        return logits
    
    @property
    def freeze_encoder(self):
        return not next(self.imu_encoder.parameters()).requires_grad
    
    def unfreeze_encoder(self):
        """Unfreeze l'encoder pour finetuning"""
        for param in self.imu_encoder.parameters():
            param.requires_grad = True


def test_models():
    """Test des modèles"""
    from config import CONFIG
    
    # Test IMU Encoder
    print("Test IMU Encoder...")
    imu_encoder = IMUEncoder(CONFIG)
    x_imu = torch.randn(4, 6, 250)  # (batch, channels, seq_len)
    cls_out, all_tokens = imu_encoder(x_imu)
    print(f"CLS output: {cls_out.shape}")
    print(f"All tokens: {all_tokens.shape}")
    
    # Test Video Encoder
    print("\nTest Video Encoder...")
    video_encoder = VideoEncoder(CONFIG)
    x_video = torch.randn(4, 10, 3, 224, 224)  # (batch, frames, C, H, W)
    video_out = video_encoder(x_video)
    print(f"Video output: {video_out.shape}")
    
    # Test Cross-Modal Model
    print("\nTest Cross-Modal Model...")
    model = CrossModalModel(CONFIG)
    imu_proj, video_proj = model(x_imu, x_video)
    print(f"IMU projection: {imu_proj.shape}")
    print(f"Video projection: {video_proj.shape}")
    
    # Test Classifier
    print("\nTest Classifier...")
    classifier = IMUClassifier(imu_encoder, CONFIG, freeze_encoder=False)
    logits = classifier(x_imu)
    print(f"Logits: {logits.shape}")


if __name__ == "__main__":
    test_models()