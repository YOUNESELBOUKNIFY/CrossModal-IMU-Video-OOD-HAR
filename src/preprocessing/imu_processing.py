import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import os
from typing import Tuple, List


class IMUPreprocessor:
    """
    Prétraitement des données IMU selon l'article :
    1. Rééchantillonnage à 50 Hz
    2. Filtrage médian (kernel=5)
    3. Normalisation z-score
    4. Extraction de fenêtres de 5 secondes (250 timestamps)
    """
    
    def __init__(self, target_freq: int = 50, window_size: int = 5):
        self.target_freq = target_freq
        self.window_size = window_size
        self.window_samples = target_freq * window_size  # 250 samples
        
    def resample_to_target_freq(self, data: np.ndarray, original_freq: int) -> np.ndarray:
        """
        Rééchantillonne les données à la fréquence cible (50 Hz)
        
        Args:
            data: Données IMU shape (n_samples, 6) [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
            original_freq: Fréquence d'échantillonnage originale
        
        Returns:
            Données rééchantillonnées à 50 Hz
        """
        if original_freq == self.target_freq:
            return data
            
        n_samples_original = len(data)
        duration = n_samples_original / original_freq
        n_samples_target = int(duration * self.target_freq)
        
        # Interpolation pour chaque canal
        resampled_data = np.zeros((n_samples_target, data.shape[1]))
        
        time_original = np.linspace(0, duration, n_samples_original)
        time_target = np.linspace(0, duration, n_samples_target)
        
        for i in range(data.shape[1]):
            interpolator = interp1d(time_original, data[:, i], kind='linear')
            resampled_data[:, i] = interpolator(time_target)
            
        return resampled_data
    
    def apply_median_filter(self, data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Applique un filtre médian pour réduire le bruit
        
        Args:
            data: Données IMU shape (n_samples, 6)
            kernel_size: Taille du noyau du filtre médian
        
        Returns:
            Données filtrées
        """
        filtered_data = np.zeros_like(data)
        
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.medfilt(data[:, i], kernel_size=kernel_size)
            
        return filtered_data
    
    def normalize_zscore(self, data: np.ndarray) -> np.ndarray:
        """
        Normalisation z-score (moyenne=0, écart-type=1)
        
        Args:
            data: Données IMU shape (n_samples, 6)
        
        Returns:
            Données normalisées
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # Éviter la division par zéro
        std[std == 0] = 1.0
        
        normalized_data = (data - mean) / std
        
        return normalized_data
    
    def extract_windows(self, data: np.ndarray, stride: int = None, 
                       overlap: bool = False) -> List[np.ndarray]:
        """
        Extrait des fenêtres de 5 secondes (250 samples à 50 Hz)
        
        Args:
            data: Données IMU prétraitées shape (n_samples, 6)
            stride: Pas de déplacement de la fenêtre (None = non-chevauchant)
            overlap: Si True, utilise un stride de window_samples//2
        
        Returns:
            Liste de fenêtres shape (250, 6)
        """
        if stride is None:
            stride = self.window_samples if not overlap else self.window_samples // 2
            
        windows = []
        n_samples = len(data)
        
        for start_idx in range(0, n_samples - self.window_samples + 1, stride):
            end_idx = start_idx + self.window_samples
            window = data[start_idx:end_idx]
            
            if len(window) == self.window_samples:
                windows.append(window)
                
        return windows
    
    def preprocess_pipeline(self, data: np.ndarray, original_freq: int, 
                           extract_windows: bool = True) -> np.ndarray or List[np.ndarray]:
        """
        Pipeline complet de prétraitement
        
        Args:
            data: Données IMU brutes shape (n_samples, 6)
            original_freq: Fréquence d'échantillonnage originale
            extract_windows: Si True, extrait les fenêtres de 5 secondes
        
        Returns:
            Données prétraitées (avec ou sans extraction de fenêtres)
        """
        print("Étape 1/4 : Rééchantillonnage à 50 Hz...")
        data = self.resample_to_target_freq(data, original_freq)
        
        print("Étape 2/4 : Filtrage médian...")
        data = self.apply_median_filter(data, kernel_size=5)
        
        print("Étape 3/4 : Normalisation z-score...")
        data = self.normalize_zscore(data)
        
        if extract_windows:
            print("Étape 4/4 : Extraction des fenêtres de 5 secondes...")
            data = self.extract_windows(data, overlap=False)
        
        print("Prétraitement terminé !")
        return data
    
    def load_and_preprocess_file(self, filepath: str, original_freq: int) -> List[np.ndarray]:
        """
        Charge et prétraite un fichier IMU
        
        Args:
            filepath: Chemin vers le fichier IMU (format CSV ou numpy)
            original_freq: Fréquence d'échantillonnage originale
        
        Returns:
            Liste de fenêtres prétraitées
        """
        # Charger les données
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath).values
        elif filepath.endswith('.npy'):
            data = np.load(filepath)
        else:
            raise ValueError("Format de fichier non supporté. Utilisez .csv ou .npy")
        
        # Vérifier que nous avons 6 canaux (3 acc + 3 gyro)
        if data.shape[1] != 6:
            raise ValueError(f"Attendu 6 canaux IMU, obtenu {data.shape[1]}")
        
        # Prétraiter
        windows = self.preprocess_pipeline(data, original_freq, extract_windows=True)
        
        return windows


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialiser le préprocesseur
    preprocessor = IMUPreprocessor(target_freq=50, window_size=5)
    
    # Créer des données synthétiques pour tester
    np.random.seed(42)
    original_freq = 25  # MMEA utilise 25 Hz
    duration = 60  # 60 secondes
    n_samples = original_freq * duration
    
    # Simuler des données IMU (acc_xyz + gyro_xyz)
    synthetic_data = np.random.randn(n_samples, 6)
    
    # Prétraiter
    windows = preprocessor.preprocess_pipeline(synthetic_data, original_freq)
    
    print(f"\nRésultats :")
    print(f"Données originales : {synthetic_data.shape}")
    print(f"Nombre de fenêtres extraites : {len(windows)}")
    print(f"Shape de chaque fenêtre : {windows[0].shape}")
    print(f"Chaque fenêtre contient {windows[0].shape[0]} timestamps et {windows[0].shape[1]} canaux")