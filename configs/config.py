"""
Configuration centrale pour le projet IMU-Video HAR
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PathConfig:
    """Configuration des chemins Kaggle/Local"""
    # Détection automatique de l'environnement
    is_kaggle: bool = os.path.exists('/kaggle')
    
    # Chemins de base
    base_input: Path = field(default_factory=lambda: 
        Path('/kaggle/input/dataset-har/UESTC-MMEA-CL') if os.path.exists('/kaggle') 
        else Path('./data/UESTC-MMEA-CL'))
    
    base_output: Path = field(default_factory=lambda: 
        Path('/kaggle/working') if os.path.exists('/kaggle') 
        else Path('./outputs'))
    
    # Splits
    train_file: str = 'train.txt'
    val_file: str = 'val.txt'
    test_file: str = 'test.txt'
    
    # Données
    sensor_dir: str = 'sensor'
    video_dir: str = 'video'
    
    def __post_init__(self):
        self.base_input = Path(self.base_input)
        self.base_output = Path(self.base_output)
        self.base_output.mkdir(parents=True, exist_ok=True)
        
        # Sous-dossiers output
        self.preprocessed_dir = self.base_output / 'preprocessed'
        self.checkpoints_dir = self.base_output / 'checkpoints'
        self.logs_dir = self.base_output / 'logs'
        self.results_dir = self.base_output / 'results'
        
        for d in [self.preprocessed_dir, self.checkpoints_dir, 
                  self.logs_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Configuration du preprocessing et des données"""
    # IMU
    imu_window_size: int = 250  # 5 secondes à 50Hz
    imu_stride: int = 125  # 50% overlap
    imu_sampling_rate: int = 50  # Hz
    imu_channels: int = 6  # 3 acc + 3 gyro
    
    # Vidéo
    video_fps: int = 25
    video_frames_per_window: int = 10  # 10 frames par clip de 5s
    video_resize: tuple = (224, 224)
    
    # Normalisation
    normalize_imu: bool = True
    median_filter_kernel: int = 5
    
    # Augmentation (optionnel)
    use_augmentation: bool = False
    jitter_strength: float = 0.1
    time_warp_strength: float = 0.2


@dataclass
class ModelConfig:
    """Configuration des modèles"""
    # IMU Encoder (PatchTST-like)
    imu_patch_size: int = 16
    imu_stride: int = 16
    imu_d_model: int = 128
    imu_nhead: int = 8
    imu_num_layers: int = 4
    imu_dropout: float = 0.1
    
    # Video Encoder (simple pour commencer)
    video_backbone: str = 'resnet18'  # ou 'mobilenet_v2' pour Kaggle
    video_pretrained: bool = True
    video_d_model: int = 512
    
    # Projection heads
    projection_dim: int = 256
    projection_hidden_dim: int = 512
    
    # Classifier
    num_classes: int = 32  # MMEA a 32 classes
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    classifier_dropout: float = 0.3


@dataclass
class TrainingConfig:
    """Configuration de l'entraînement"""
    # General
    seed: int = 42
    device: str = 'cuda'  # ou 'cpu'
    num_workers: int = 2
    
    # Cross-modal pretraining
    pretrain_epochs: int = 50
    pretrain_batch_size: int = 32
    pretrain_lr: float = 1e-4
    pretrain_weight_decay: float = 0.01
    pretrain_warmup_epochs: int = 5
    
    # Contrastive loss
    temperature: float = 0.07
    use_sigmoid_loss: bool = True  # Comme dans l'article
    
    # Classification
    train_epochs: int = 100
    train_batch_size: int = 64
    train_lr_encoder: float = 1e-6  # pour finetuning
    train_lr_head: float = 1e-3     # pour classification head
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001
    
    # Checkpointing
    save_every: int = 5
    save_best_only: bool = True


@dataclass
class EvalConfig:
    """Configuration de l'évaluation"""
    # Métrics
    metrics: List[str] = field(default_factory=lambda: 
        ['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro'])
    
    # Few-shot settings (comme l'article)
    few_shot_samples: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    few_shot_runs: int = 5  # nombre de runs avec différents subsets
    
    # Modes d'évaluation
    eval_modes: List[str] = field(default_factory=lambda: 
        ['linear_probe', 'finetune'])


class Config:
    """Configuration globale"""
    def __init__(self):
        self.paths = PathConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.eval = EvalConfig()
    
    def to_dict(self):
        """Export config en dict pour sauvegarde"""
        return {
            'paths': vars(self.paths),
            'data': vars(self.data),
            'model': vars(self.model),
            'training': vars(self.training),
            'eval': vars(self.eval)
        }
    
    def save(self, path: str):
        """Sauvegarde la config"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str):
        """Charge une config"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        # TODO: reconstruction depuis dict si besoin
        return cls()


# Instance globale
CONFIG = Config()