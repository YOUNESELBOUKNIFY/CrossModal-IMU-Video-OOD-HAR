import cv2
import numpy as np
import os
from typing import List, Tuple
from pathlib import Path


class VideoPreprocessor:
    """
    Prétraitement des vidéos selon l'article :
    1. Extraire 10 frames d'un segment de 5 secondes
    2. Diviser les frames en 10 chunks égaux
    3. Sélectionner aléatoirement 1 frame par chunk
    """
    
    def __init__(self, target_fps: int = 25, segment_duration: int = 5, 
                 num_frames: int = 10, img_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            target_fps: FPS de la vidéo (MMEA = 25 fps)
            segment_duration: Durée du segment en secondes
            num_frames: Nombre de frames à extraire par segment
            img_size: Taille cible pour redimensionner les images
        """
        self.target_fps = target_fps
        self.segment_duration = segment_duration
        self.num_frames = num_frames
        self.img_size = img_size
        self.frames_per_segment = target_fps * segment_duration  # 125 frames pour 5 sec
        
    def extract_frames_from_video(self, video_path: str) -> List[np.ndarray]:
        """
        Extrait toutes les frames d'une vidéo
        
        Args:
            video_path: Chemin vers le fichier vidéo
        
        Returns:
            Liste de frames (H, W, 3)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convertir BGR (OpenCV) en RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Redimensionne une frame à la taille cible
        
        Args:
            frame: Image shape (H, W, 3)
        
        Returns:
            Image redimensionnée shape (224, 224, 3)
        """
        resized = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_LINEAR)
        return resized
    
    def sample_frames_from_segment(self, frames: List[np.ndarray], 
                                   random_sampling: bool = True,
                                   seed: int = None) -> np.ndarray:
        """
        Échantillonne 10 frames d'un segment de 5 secondes
        Stratégie : diviser en 10 chunks et sélectionner 1 frame par chunk
        
        Args:
            frames: Liste de frames du segment (125 frames pour 5 sec à 25 fps)
            random_sampling: Si True, sélection aléatoire; sinon, prendre le milieu
            seed: Graine pour la reproductibilité
        
        Returns:
            Array de frames shape (10, 224, 224, 3)
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_frames = len(frames)
        chunk_size = n_frames // self.num_frames
        
        sampled_frames = []
        
        for i in range(self.num_frames):
            chunk_start = i * chunk_size
            chunk_end = chunk_start + chunk_size if i < self.num_frames - 1 else n_frames
            
            if random_sampling:
                # Sélectionner aléatoirement dans le chunk
                frame_idx = np.random.randint(chunk_start, chunk_end)
            else:
                # Prendre la frame du milieu du chunk
                frame_idx = (chunk_start + chunk_end) // 2
            
            frame = frames[frame_idx]
            frame_resized = self.resize_frame(frame)
            sampled_frames.append(frame_resized)
        
        # Convertir en array numpy (10, 224, 224, 3)
        sampled_frames = np.stack(sampled_frames, axis=0)
        
        return sampled_frames
    
    def normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Normalise les frames pour les réseaux de neurones
        Normalisation ImageNet : mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        
        Args:
            frames: Array de frames shape (10, 224, 224, 3), valeurs [0, 255]
        
        Returns:
            Frames normalisées shape (10, 3, 224, 224), valeurs normalisées
        """
        # Convertir en float et normaliser à [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Normalisation ImageNet
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        
        frames = (frames - mean) / std
        
        # Transposer pour avoir le format (10, 3, 224, 224) - format PyTorch
        frames = np.transpose(frames, (0, 3, 1, 2))
        
        return frames
    
    def extract_video_segments(self, video_path: str, 
                              imu_windows_count: int,
                              random_sampling: bool = True) -> List[np.ndarray]:
        """
        Extrait des segments vidéo alignés avec les fenêtres IMU
        
        Args:
            video_path: Chemin vers le fichier vidéo
            imu_windows_count: Nombre de fenêtres IMU (pour aligner)
            random_sampling: Si True, échantillonnage aléatoire des frames
        
        Returns:
            Liste de segments vidéo, chaque segment shape (10, 3, 224, 224)
        """
        # Extraire toutes les frames
        all_frames = self.extract_frames_from_video(video_path)
        total_frames = len(all_frames)
        
        print(f"Vidéo chargée : {total_frames} frames")
        
        video_segments = []
        
        # Extraire les segments
        for i in range(imu_windows_count):
            start_idx = i * self.frames_per_segment
            end_idx = start_idx + self.frames_per_segment
            
            if end_idx > total_frames:
                print(f"Attention : segment {i} dépasse la longueur de la vidéo")
                break
            
            segment_frames = all_frames[start_idx:end_idx]
            
            # Échantillonner 10 frames
            sampled_frames = self.sample_frames_from_segment(
                segment_frames, 
                random_sampling=random_sampling,
                seed=i  # Pour la reproductibilité
            )
            
            # Normaliser
            normalized_frames = self.normalize_frames(sampled_frames)
            
            video_segments.append(normalized_frames)
        
        print(f"Nombre de segments vidéo extraits : {len(video_segments)}")
        
        return video_segments
    
    def preprocess_paired_data(self, video_path: str, 
                              imu_windows: List[np.ndarray],
                              random_sampling: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Prétraite les données vidéo-IMU appariées
        
        Args:
            video_path: Chemin vers le fichier vidéo
            imu_windows: Liste de fenêtres IMU prétraitées
            random_sampling: Si True, échantillonnage aléatoire des frames vidéo
        
        Returns:
            Tuple (video_segments, imu_windows) alignés
        """
        video_segments = self.extract_video_segments(
            video_path, 
            len(imu_windows),
            random_sampling
        )
        
        # S'assurer que les longueurs correspondent
        min_length = min(len(video_segments), len(imu_windows))
        video_segments = video_segments[:min_length]
        imu_windows = imu_windows[:min_length]
        
        print(f"\nDonnées appariées : {min_length} paires IMU-Vidéo")
        
        return video_segments, imu_windows


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialiser le préprocesseur
    video_processor = VideoPreprocessor(
        target_fps=25,
        segment_duration=5,
        num_frames=10,
        img_size=(224, 224)
    )
    
    # Exemple avec des données synthétiques
    print("Test avec données synthétiques...")
    
    # Créer des frames synthétiques (125 frames pour 5 secondes à 25 fps)
    synthetic_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                       for _ in range(125)]
    
    # Échantillonner 10 frames
    sampled_frames = video_processor.sample_frames_from_segment(
        synthetic_frames, 
        random_sampling=False
    )
    
    print(f"Frames échantillonnées : {sampled_frames.shape}")
    
    # Normaliser
    normalized_frames = video_processor.normalize_frames(sampled_frames)
    
    print(f"Frames normalisées : {normalized_frames.shape}")
    print(f"Range des valeurs : [{normalized_frames.min():.2f}, {normalized_frames.max():.2f}]")