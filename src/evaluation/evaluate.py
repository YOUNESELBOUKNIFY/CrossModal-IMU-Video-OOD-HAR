import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from pathlib import Path


class IMUVideoDataset(Dataset):
    """
    Dataset pour les paires IMU-Vidéo
    """
    def __init__(self, imu_windows, video_segments, labels=None):
        """
        Args:
            imu_windows: Liste de fenêtres IMU shape (250, 6)
            video_segments: Liste de segments vidéo shape (10, 3, 224, 224)
            labels: Labels optionnels pour les activités
        """
        assert len(imu_windows) == len(video_segments), \
            "Le nombre de fenêtres IMU doit correspondre au nombre de segments vidéo"
        
        self.imu_windows = imu_windows
        self.video_segments = video_segments
        self.labels = labels
        
    def __len__(self):
        return len(self.imu_windows)
    
    def __getitem__(self, idx):
        imu = torch.FloatTensor(self.imu_windows[idx]).transpose(0, 1)  # (6, 250)
        video = torch.FloatTensor(self.video_segments[idx])  # (10, 3, 224, 224)
        
        if self.labels is not None:
            label = self.labels[idx]
            return imu, video, label
        
        return imu, video


class PretrainConfig:
    """Configuration pour le pretraining"""
    def __init__(self):
        # Données
        self.batch_size = 32
        self.num_workers = 4
        
        # Architecture IMU Encoder
        self.imu_seq_len = 250
        self.imu_patch_size = 16
        self.imu_stride = 16
        self.imu_num_channels = 6
        self.imu_embed_dim = 128
        self.imu_num_heads = 8
        self.imu_num_layers = 3
        
        # Architecture Video Encoder
        self.video_num_frames = 10
        self.video_embed_dim = 512
        
        # Projection
        self.projection_dim = 256
        
        # Entraînement
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.warmup_epochs = 5
        
        # Loss
        self.loss_init_temp = 10.0
        self.loss_init_bias = -10.0
        
        # Sauvegarde
        self.checkpoint_dir = "checkpoints"
        self.save_frequency = 5
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrossModalPretrainer:
    """
    Classe pour gérer le pretraining cross-modal IMU-Vidéo
    """
    def __init__(self, imu_encoder, video_encoder, loss_fn, config):
        self.imu_encoder = imu_encoder.to(config.device)
        self.video_encoder = video_encoder.to(config.device)
        self.loss_fn = loss_fn.to(config.device)
        self.config = config
        
        # Optimizer
        params = list(imu_encoder.parameters()) + \
                 list(video_encoder.parameters()) + \
                 list(loss_fn.parameters())
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler avec warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs - config.warmup_epochs,
            eta_min=1e-6
        )
        
        # Tracking
        self.train_losses = []
        self.current_epoch = 0
        
        # Créer le répertoire de sauvegarde
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def warmup_lr(self, epoch, warmup_epochs):
        """Learning rate warmup"""
        if epoch < warmup_epochs:
            lr = self.config.learning_rate * (epoch + 1) / warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def train_epoch(self, dataloader):
        """Entraîne le modèle pendant une époque"""
        self.imu_encoder.train()
        self.video_encoder.train()
        
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            imu_data, video_data = batch[0], batch[1]
            imu_data = imu_data.to(self.config.device)
            video_data = video_data.to(self.config.device)
            
            # Forward pass
            imu_projection = self.imu_encoder(imu_data, return_projection=True)
            video_projection = self.video_encoder(video_data, return_projection=True)
            
            # Calculer la loss
            loss = self.loss_fn(imu_projection, video_projection)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.imu_encoder.parameters()) + 
                list(self.video_encoder.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # Tracking
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        self.train_losses.append(avg_epoch_loss)
        
        return avg_epoch_loss
    
    def train(self, train_dataloader, val_dataloader=None):
        """
        Boucle d'entraînement principale
        
        Args:
            train_dataloader: DataLoader pour l'entraînement
            val_dataloader: DataLoader pour la validation (optionnel)
        """
        print(f"\nDébut du pretraining sur {self.config.device}")
        print(f"Nombre d'époques : {self.config.num_epochs}")
        print(f"Batch size : {self.config.batch_size}")
        print(f"Learning rate : {self.config.learning_rate}")
        print("="*60)
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Warmup
            if epoch < self.config.warmup_epochs:
                self.warmup_lr(epoch, self.config.warmup_epochs)
            
            # Entraîner
            train_loss = self.train_epoch(train_dataloader)
            
            # Validation (optionnel)
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs} - "
                      f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs} - "
                      f"Train Loss: {train_loss:.4f}")
            
            # Learning rate scheduler (après warmup)
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            # Sauvegarder le checkpoint
            if (epoch + 1) % self.config.save_frequency == 0:
                self.save_checkpoint(epoch + 1)
        
        # Sauvegarder le modèle final
        self.save_checkpoint("final")
        print("\nPretraining terminé !")
    
    def validate(self, val_dataloader):
        """Évalue le modèle sur les données de validation"""
        self.imu_encoder.eval()
        self.video_encoder.eval()
        
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_dataloader:
                imu_data, video_data = batch[0], batch[1]
                imu_data = imu_data.to(self.config.device)
                video_data = video_data.to(self.config.device)
                
                # Forward
                imu_projection = self.imu_encoder(imu_data, return_projection=True)
                video_projection = self.video_encoder(video_data, return_projection=True)
                
                # Loss
                loss = self.loss_fn(imu_projection, video_projection)
                val_loss += loss.item()
        
        return val_loss / len(val_dataloader)
    
    def save_checkpoint(self, epoch):
        """Sauvegarde un checkpoint"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pt"
        )
        
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'imu_encoder_state_dict': self.imu_encoder.state_dict(),
            'video_encoder_state_dict': self.video_encoder.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'config': vars(self.config)
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint sauvegardé : {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Charge un checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.imu_encoder.load_state_dict(checkpoint['imu_encoder_state_dict'])
        self.video_encoder.load_state_dict(checkpoint['video_encoder_state_dict'])
        self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.current_epoch = checkpoint['epoch']
        
        print(f"Checkpoint chargé : {checkpoint_path}")
        print(f"Reprise à l'époque {self.current_epoch}")


# Exemple d'utilisation
if __name__ == "__main__":
    from models.imu_encoder import IMUEncoder
    from models.video_encoder import SimpleVideoEncoder
    from models.contrastive_loss import SigmoidContrastiveLoss
    
    # Configuration
    config = PretrainConfig()
    config.num_epochs = 5  # Réduit pour le test
    config.batch_size = 4
    
    # Créer des données synthétiques
    num_samples = 100
    imu_windows = [np.random.randn(250, 6) for _ in range(num_samples)]
    video_segments = [np.random.randn(10, 3, 224, 224) for _ in range(num_samples)]
    
    # Dataset et DataLoader
    dataset = IMUVideoDataset(imu_windows, video_segments)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Modèles
    imu_encoder = IMUEncoder(
        seq_len=config.imu_seq_len,
        patch_size=config.imu_patch_size,
        stride=config.imu_stride,
        num_channels=config.imu_num_channels,
        embed_dim=config.imu_embed_dim,
        projection_dim=config.projection_dim
    )
    
    video_encoder = SimpleVideoEncoder(
        num_frames=config.video_num_frames,
        embed_dim=config.video_embed_dim,
        projection_dim=config.projection_dim
    )
    
    loss_fn = SigmoidContrastiveLoss(
        init_temp=config.loss_init_temp,
        init_bias=config.loss_init_bias
    )
    
    # Pretrainer
    pretrainer = CrossModalPretrainer(imu_encoder, video_encoder, loss_fn, config)
    
    # Entraîner
    pretrainer.train(dataloader)