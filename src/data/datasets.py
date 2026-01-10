"""
PyTorch Datasets pour IMU-Video Cross-modal Learning
Conforme à l'article avec gestion robuste des vidéos manquantes
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import cv2
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')


class CrossModalDataset(Dataset):
    """
    Dataset pour l'entraînement cross-modal IMU-Video
    Retourne: {'imu': tensor, 'video': tensor, 'idx': int}
    
    Gestion automatique des vidéos manquantes (frames noires)
    """
    
    def __init__(self, 
                 metadata_df: pd.DataFrame,
                 config,
                 video_transform=None,
                 return_paths=False):
        """
        Args:
            metadata_df: DataFrame avec colonnes [imu_window_path, video_path, start_frame, ...]
            config: Config object
            video_transform: transformations pour vidéo (si None, utilise default)
            return_paths: si True, retourne aussi les chemins (pour debug)
        """
        self.df = metadata_df.copy()
        self.config = config
        self.data_cfg = config.data
        self.paths_cfg = config.paths
        self.return_paths = return_paths
        
        # Filtrer les samples avec vidéo si nécessaire
        # (optionnel: enlever ce filtre pour garder tous les samples)
        # self.df = self.df[self.df['video_exists'] == True]
        
        print(f"Dataset créé avec {len(self.df)} samples")
        if 'video_exists' in self.df.columns:
            n_with_video = self.df['video_exists'].sum()
            print(f"  - Avec vidéo: {n_with_video}")
            print(f"  - Sans vidéo: {len(self.df) - n_with_video}")
        
        # Video transforms
        if video_transform is None:
            self.video_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.data_cfg.video_resize),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet stats
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.video_transform = video_transform
    
    def __len__(self):
        return len(self.df)
    
    def load_imu_window(self, imu_path: str) -> torch.Tensor:
        """
        Charge un window IMU prétraité
        Returns: tensor shape (channels, window_size) pour Conv1D
        """
        try:
            imu_data = np.load(imu_path)  # Shape: (window_size, channels)
            
            # Vérifier dimensions
            if imu_data.shape != (self.data_cfg.imu_window_size, self.data_cfg.imu_channels):
                print(f"Avertissement: shape IMU inattendu {imu_data.shape}, "
                      f"attendu ({self.data_cfg.imu_window_size}, {self.data_cfg.imu_channels})")
            
            # Convertir en tensor et transposer pour Conv1D: (channels, window_size)
            imu_tensor = torch.FloatTensor(imu_data).transpose(0, 1)
            
            return imu_tensor
        
        except Exception as e:
            print(f"Erreur lors du chargement de {imu_path}: {e}")
            # Retourner tensor de zeros
            return torch.zeros(self.data_cfg.imu_channels, self.data_cfg.imu_window_size)
    
    def load_video_clip(self, video_path: str, start_frame: int) -> torch.Tensor:
        """
        Charge un clip vidéo de N frames
        Conforme à l'article: 10 frames extraites uniformément sur 5 secondes
        
        Args:
            video_path: chemin vers la vidéo
            start_frame: frame de départ
        
        Returns:
            tensor shape (num_frames, C, H, W)
        """
        full_video_path = self.paths_cfg.base_input / video_path
        
        # Si vidéo n'existe pas, retourner frames noires
        if not full_video_path.exists():
            num_frames = self.data_cfg.video_frames_per_window
            H, W = self.data_cfg.video_resize
            return torch.zeros(num_frames, 3, H, W)
        
        try:
            cap = cv2.VideoCapture(str(full_video_path))
            
            if not cap.isOpened():
                raise RuntimeError(f"Impossible d'ouvrir la vidéo: {full_video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculer les indices de frames à extraire
            # Sur 5 secondes de vidéo, extraire 10 frames uniformément
            window_duration_seconds = self.data_cfg.imu_window_size / self.data_cfg.imu_sampling_rate  # 5s
            window_duration_frames = int(window_duration_seconds * fps)
            
            # Indices des frames à extraire (10 frames espacées uniformément)
            target_frames = self.data_cfg.video_frames_per_window
            
            if window_duration_frames > 0:
                frame_indices = np.linspace(
                    start_frame,
                    start_frame + window_duration_frames - 1,
                    target_frames,
                    dtype=int
                )
            else:
                frame_indices = np.arange(start_frame, start_frame + target_frames, dtype=int)
            
            # Clip les indices pour éviter de dépasser la vidéo
            frame_indices = np.clip(frame_indices, 0, total_frames - 1)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # BGR -> RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Appliquer transforms
                    frame_tensor = self.video_transform(frame_rgb)
                    frames.append(frame_tensor)
                else:
                    # Frame noire si échec
                    H, W = self.data_cfg.video_resize
                    frames.append(torch.zeros(3, H, W))
            
            cap.release()
            
            # Si pas assez de frames, compléter avec des frames noires
            while len(frames) < target_frames:
                H, W = self.data_cfg.video_resize
                frames.append(torch.zeros(3, H, W))
            
            # Stack frames: (num_frames, C, H, W)
            video_tensor = torch.stack(frames)
            
            return video_tensor
        
        except Exception as e:
            print(f"Erreur lors du chargement de la vidéo {video_path}: {e}")
            # Retourner frames noires
            num_frames = self.data_cfg.video_frames_per_window
            H, W = self.data_cfg.video_resize
            return torch.zeros(num_frames, 3, H, W)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load IMU
        imu_tensor = self.load_imu_window(row['imu_window_path'])
        
        # Load Video
        start_frame = int(row['start_frame']) if 'start_frame' in row else 0
        video_tensor = self.load_video_clip(row['video_path'], start_frame)
        
        output = {
            'imu': imu_tensor,
            'video': video_tensor,
            'idx': idx
        }
        
        if self.return_paths:
            output['imu_path'] = row['imu_window_path']
            output['video_path'] = row['video_path']
        
        return output


class IMUClassificationDataset(Dataset):
    """
    Dataset pour classification d'activités avec IMU seul
    Retourne: {'imu': tensor, 'label': int, 'idx': int}
    """
    
    def __init__(self, 
                 metadata_df: pd.DataFrame,
                 config,
                 return_info=False):
        """
        Args:
            metadata_df: DataFrame avec colonnes [imu_window_path, label, ...]
            config: Config object
            return_info: si True, retourne aussi class_name et user_id
        """
        self.df = metadata_df.copy()
        self.config = config
        self.return_info = return_info
        
        print(f"Classification dataset créé avec {len(self.df)} samples")
        print(f"  - Classes: {self.df['label'].nunique()}")
        
    def __len__(self):
        return len(self.df)
    
    def load_imu_window(self, imu_path: str) -> torch.Tensor:
        """Charge un window IMU"""
        try:
            imu_data = np.load(imu_path)
            imu_tensor = torch.FloatTensor(imu_data).transpose(0, 1)
            return imu_tensor
        except Exception as e:
            print(f"Erreur chargement {imu_path}: {e}")
            return torch.zeros(self.config.data.imu_channels, 
                             self.config.data.imu_window_size)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        imu_tensor = self.load_imu_window(row['imu_window_path'])
        label = int(row['label'])
        
        output = {
            'imu': imu_tensor,
            'label': label,
            'idx': idx
        }
        
        if self.return_info:
            output['class_name'] = row['class_name']
            output['user_id'] = row['user_id']
        
        return output


class FewShotSampler:
    """
    Sampler pour créer des subsets few-shot
    Utilisé pour les expériences avec 10, 20, 50, 100 samples par classe
    """
    
    def __init__(self, metadata_df: pd.DataFrame, config):
        self.df = metadata_df
        self.config = config
        
    def sample_k_per_class(self, k: int, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Sample k exemples par classe de manière équilibrée
        
        Args:
            k: nombre d'exemples par classe
            seed: random seed pour reproductibilité
        
        Returns:
            DataFrame subset avec k samples par classe
        """
        if seed is not None:
            np.random.seed(seed)
        
        sampled = []
        
        for class_name in self.df['class_name'].unique():
            class_df = self.df[self.df['class_name'] == class_name]
            
            if len(class_df) >= k:
                # Si assez de samples, en prendre k aléatoirement
                subset = class_df.sample(n=k, random_state=seed)
            else:
                # Si moins de k samples, prendre tous + warning
                print(f"Avertissement: classe '{class_name}' a seulement "
                      f"{len(class_df)} samples (< {k})")
                subset = class_df
            
            sampled.append(subset)
        
        result_df = pd.concat(sampled, ignore_index=True)
        
        print(f"Few-shot sampler: {len(result_df)} samples "
              f"({k} par classe × {self.df['class_name'].nunique()} classes)")
        
        return result_df
    
    def sample_balanced_test_set(self, n_per_class: int = 20, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Crée un test set équilibré avec n_per_class samples par classe
        Utilisé pour l'évaluation few-shot
        """
        return self.sample_k_per_class(n_per_class, seed)


def create_dataloaders(config, 
                       train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       mode: str = 'cross_modal',
                       shuffle_train: bool = True) -> Dict[str, DataLoader]:
    """
    Crée les dataloaders pour train/val/test
    
    Args:
        config: Config object
        train_df, val_df, test_df: DataFrames metadata
        mode: 'cross_modal' ou 'classification'
        shuffle_train: si True, shuffle le train set
    
    Returns:
        dict avec clés 'train', 'val', 'test' contenant les dataloaders
    """
    
    if mode == 'cross_modal':
        print("Création des dataloaders cross-modal...")
        train_dataset = CrossModalDataset(train_df, config)
        val_dataset = CrossModalDataset(val_df, config)
        test_dataset = CrossModalDataset(test_df, config)
        batch_size = config.training.pretrain_batch_size
        
    elif mode == 'classification':
        print("Création des dataloaders classification...")
        train_dataset = IMUClassificationDataset(train_df, config)
        val_dataset = IMUClassificationDataset(val_df, config)
        test_dataset = IMUClassificationDataset(test_df, config)
        batch_size = config.training.train_batch_size
    
    else:
        raise ValueError(f"Mode inconnu: {mode}")
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Drop last batch si incomplet
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"✓ Dataloaders créés:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def get_class_weights(metadata_df: pd.DataFrame) -> torch.Tensor:
    """
    Calcule les poids de classe pour gérer le déséquilibre
    Utile si vous voulez utiliser une weighted loss
    
    Returns:
        tensor de poids shape (num_classes,)
    """
    class_counts = metadata_df['label'].value_counts().sort_index()
    total = len(metadata_df)
    
    # Inverse frequency weighting
    weights = total / (len(class_counts) * class_counts.values)
    
    return torch.FloatTensor(weights)


def test_dataloaders():
    """Test des dataloaders avec quelques samples"""
    import sys
    sys.path.append('.')
    from config import CONFIG
    
    print("\n" + "="*60)
    print("TEST DES DATALOADERS")
    print("="*60)
    
    # Charger metadata
    try:
        train_df = pd.read_csv(CONFIG.paths.preprocessed_dir / 'train_metadata.csv')
        print(f"✓ Train metadata chargée: {len(train_df)} samples")
    except FileNotFoundError:
        print("❌ Metadata non trouvée. Veuillez d'abord exécuter le preprocessing.")
        return
    
    # Test 1: CrossModalDataset
    print("\n--- Test 1: CrossModalDataset ---")
    dataset = CrossModalDataset(train_df.head(10), CONFIG, return_paths=True)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample 0:")
    print(f"  IMU shape: {sample['imu'].shape}")
    print(f"  Video shape: {sample['video'].shape}")
    print(f"  IMU path: {sample['imu_path']}")
    print(f"  Video path: {sample['video_path']}")
    
    # Test 2: IMUClassificationDataset
    print("\n--- Test 2: IMUClassificationDataset ---")
    dataset_clf = IMUClassificationDataset(train_df.head(10), CONFIG, return_info=True)
    sample_clf = dataset_clf[0]
    print(f"Sample 0:")
    print(f"  IMU shape: {sample_clf['imu'].shape}")
    print(f"  Label: {sample_clf['label']}")
    print(f"  Class name: {sample_clf['class_name']}")
    
    # Test 3: Dataloader
    print("\n--- Test 3: Dataloader ---")
    loader = DataLoader(dataset_clf, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print(f"Batch:")
    print(f"  IMU shape: {batch['imu'].shape}")
    print(f"  Labels shape: {batch['label'].shape}")
    print(f"  Labels: {batch['label'].tolist()}")
    
    # Test 4: FewShotSampler
    print("\n--- Test 4: FewShotSampler ---")
    sampler = FewShotSampler(train_df, CONFIG)
    subset = sampler.sample_k_per_class(k=5, seed=42)
    print(f"Subset size: {len(subset)}")
    print(f"Classes: {subset['class_name'].value_counts()}")
    
    # Test 5: Class weights
    print("\n--- Test 5: Class Weights ---")
    weights = get_class_weights(train_df)
    print(f"Weights shape: {weights.shape}")
    print(f"Top 5 weights: {weights[:5].tolist()}")
    
    print("\n✓ Tous les tests réussis!")


if __name__ == "__main__":
    test_dataloaders()