import torch
from torch.utils.data import Dataset
import numpy as np
import os
# Import de vos classes de prétraitement
from src.preprocessing.imu_processing import IMUPreprocessor
from src.preprocessing.video_processing import VideoPreprocessor

class MMEADataset(Dataset):
    def __init__(self, data_pairs, mode='pretrain', imu_params=None, video_params=None):
        """
        data_pairs: Liste de tuples [(path_imu, path_video, label), ...]
        mode: 'pretrain' (retourne paire IMU-Vidéo) ou 'finetune' (retourne IMU + Label)
        """
        self.data_pairs = data_pairs
        self.mode = mode
        
        # Initialisation des préprocesseurs
        self.imu_proc = IMUPreprocessor(**(imu_params or {}))
        self.video_proc = VideoPreprocessor(**(video_params or {}))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        path_imu, path_video, label = self.data_pairs[idx]
        
        # 1. Charger et traiter l'IMU
        # Note: Dans une implémentation réelle, pour la rapidité, on charge souvent 
        # des données déjà prétraitées (.npy) plutôt que de traiter le CSV à la volée.
        try:
            # Simulation du chargement (à adapter selon votre format de stockage réel)
            if path_imu.endswith('.npy'):
                 imu_window = np.load(path_imu)
            else:
                 # Fallback ou traitement complet si nécessaire
                 imu_window = np.random.randn(250, 6) # Placeholder
            
            # Conversion en Tensor [Channels, Time]
            imu_tensor = torch.FloatTensor(imu_window).transpose(0, 1) # (6, 250)
            
        except Exception as e:
            print(f"Erreur chargement IMU {path_imu}: {e}")
            imu_tensor = torch.zeros(6, 250)

        # Si on est en mode finetune, on n'a pas besoin de la vidéo
        if self.mode == 'finetune':
            return imu_tensor, torch.tensor(label, dtype=torch.long)

        # 2. Charger et traiter la Vidéo (Seulement pour pretrain)
        try:
            # Ici on utiliserait video_proc.extract_video_segments ou similaire
            # Pour l'exemple, supposons que nous avons déjà les frames extraites ou on le fait à la volée
            # video_segments = self.video_proc.extract_video_segments(path_video, 1)
            # video_tensor = torch.FloatTensor(video_segments[0]) 
            
            # Placeholder pour que le code tourne sans les fichiers lourds
            video_tensor = torch.randn(10, 3, 224, 224) 
            
        except Exception as e:
            print(f"Erreur chargement Vidéo {path_video}: {e}")
            video_tensor = torch.zeros(10, 3, 224, 224)

        return imu_tensor, video_tensor