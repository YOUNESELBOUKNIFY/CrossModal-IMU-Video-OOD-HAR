import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .imu_encoder import IMUEncoder
from .video_encoder import VideoEncoder

class CrossModalContrastiveModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 1. Initialisation des encodeurs (basé sur vos fichiers précédents)
        self.imu_encoder = IMUEncoder(
            seq_len=cfg['model']['imu']['seq_len'],
            num_channels=cfg['model']['imu']['channels'],
            embed_dim=cfg['model']['imu']['embed_dim'],
            projection_dim=cfg['model']['projection_dim']
        )
        
        self.video_encoder = VideoEncoder(
            num_frames=cfg['model']['video']['num_frames'],
            embed_dim=cfg['model']['video']['embed_dim'],
            projection_dim=cfg['model']['projection_dim'],
            pretrained=cfg['model']['video']['pretrained']
        )
        
        # 2. Paramètre de température apprenable (comme dans CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, imu_data, video_data):
        # Extraction des caractéristiques projetées
        # imu_features: [batch_size, projection_dim]
        # video_features: [batch_size, projection_dim]
        imu_features = self.imu_encoder(imu_data, return_projection=True)
        video_features = self.video_encoder(video_data, return_projection=True)
        
        # Normalisation (L2 norm) pour le calcul de similarité cosinus
        imu_features = F.normalize(imu_features, dim=1)
        video_features = F.normalize(video_features, dim=1)
        
        return imu_features, video_features

    def compute_loss(self, imu_features, video_features):
        """
        Calcule la perte InfoNCE (Contrastive Loss)
        """
        # Récupérer la température
        logit_scale = self.logit_scale.exp()
        
        # Calcul de la matrice de similarité : (Batch x Batch)
        logits_per_imu = logit_scale * imu_features @ video_features.t()
        logits_per_video = logits_per_imu.t()
        
        # Les labels sont la diagonale (l'élément i de l'IMU correspond à l'élément i de la Vidéo)
        batch_size = imu_features.shape[0]
        labels = torch.arange(batch_size, device=imu_features.device, dtype=torch.long)
        
        # Perte symétrique (IMU->Video et Video->IMU)
        loss_i2v = F.cross_entropy(logits_per_imu, labels)
        loss_v2i = F.cross_entropy(logits_per_video, labels)
        
        total_loss = (loss_i2v + loss_v2i) / 2
        return total_loss