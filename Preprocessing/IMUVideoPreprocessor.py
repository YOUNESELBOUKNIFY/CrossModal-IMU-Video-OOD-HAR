import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
import random

class IMUVideoPreprocessing:
    """
    Préprocessing fidèle à l'article "IMU-Video Cross-modal Self-supervision for HAR".

    Attributs (variables de classe) :
    - target_freq (int):
        Quoi : fréquence d'échantillonnage IMU cible après rééchantillonnage (par défaut 50 Hz).
        Pourquoi : l’article impose 50 Hz pour uniformiser Ego4D / MMEA / PD et rendre les modèles comparables.
        (Upsampling ou downsampling selon la fréquence d'origine).

    - window_sec (int/float):
        Quoi : durée d’une fenêtre temporelle IMU en secondes (par défaut 5 s).
        Pourquoi : l’article entraîne/évalue sur des fenêtres fixes de 5 secondes. 

    - window_size (int):
        Quoi : nombre d’échantillons IMU par fenêtre = target_freq * window_sec.
              Exemple : 50 Hz * 5 s = 250 points.
        Pourquoi : correspond exactement au “context length = 250” (fenêtre 5 s à 50 Hz). 

    - num_video_frames (int):
        Quoi : nombre d’images (frames) représentant un segment vidéo de 5 s (par défaut 10).
        Pourquoi : l’article divise les frames de 5 s en 10 chunks et choisit 1 frame aléatoire par chunk
                  => 10 frames qui résument le “flow” de l’action. 

    - median_kernel (int):
        Quoi : taille du noyau du filtre médian (par défaut 5).
        Pourquoi : l’article applique un median filtering kernel size 5 pour supprimer le bruit IMU. 
        Note : typiquement kernel impair (3,5,7...) sinon certains filtres refusent.

    - scaler (StandardScaler):
        Quoi : objet qui applique la normalisation z-score (mean=0, std=1).
        Pourquoi : l’article normalise les signaux IMU avec z-score après filtrage. 
        Bon usage pratique : on “fit” le scaler sur le train seulement, puis “transform” val/test (pour éviter leakage).

    - seed (int) + random.seed / np.random.seed:
        Quoi : graine aléatoire pour reproductibilité.
        Pourquoi : ton sampling aléatoire des frames (1 frame par chunk) dépend du hasard ;
                  fixer un seed permet de reproduire exactement les mêmes fenêtres/frames d’un run à l’autre.
    """

    def __init__(self,
                 target_freq=50,
                 window_sec=5,
                 num_video_frames=10,
                 median_kernel=5,
                 seed=42):
        self.target_freq = target_freq
        self.window_sec = window_sec
        self.window_size = target_freq * window_sec  # 250
        self.num_video_frames = num_video_frames
        self.median_kernel = median_kernel
        self.scaler = StandardScaler()

        random.seed(seed)
        np.random.seed(seed)

    # -------------------------------------------------
    # IMU preprocessing (Section 3.2 – Preprocessing)
    # -------------------------------------------------
    def preprocess_imu(self, imu_df: pd.DataFrame, original_freq: float):
        """
        imu_df: DataFrame [T, 6] -> acc(x,y,z), gyro(x,y,z)
        """
        imu = imu_df.values.astype(np.float32)

        # 1) Resampling to 50 Hz
        num_samples = int(len(imu) * self.target_freq / original_freq)
        imu = signal.resample(imu, num_samples, axis=0)

        # 2) Median filtering (kernel size = 5)
        imu = signal.medfilt(imu, kernel_size=(self.median_kernel, 1))

        # 3) Z-score normalization (global)
        imu = self.scaler.fit_transform(imu)

        return imu

    # -------------------------------------------------
    # Windowing + video frame selection
    # -------------------------------------------------
    def extract_windows(self, imu_data, video_frames=None):
        """
        imu_data: np.ndarray [T, 6]
        video_frames: list of frame indices or paths (aligned with imu_data)
        """
        imu_windows = []
        video_windows = []

        T = len(imu_data)

        # 4) Non-overlapping 5-second windows
        for start in range(0, T - self.window_size + 1, self.window_size):
            end = start + self.window_size
            imu_windows.append(imu_data[start:end])

            # 5) Video frame sampling (only for cross-modal SSL)
            if video_frames is not None:
                frames_5s = video_frames[start:end]

                chunk_size = len(frames_5s) // self.num_video_frames
                selected_frames = []

                for i in range(self.num_video_frames):
                    c_start = i * chunk_size
                    c_end = (i + 1) * chunk_size
                    idx = random.randint(c_start, c_end - 1)
                    selected_frames.append(frames_5s[idx])

                video_windows.append(selected_frames)

        return np.array(imu_windows), np.array(video_windows)

# =====================================================
# Exemple d’utilisation (simulation fidèle article)
# =====================================================

if __name__ == "__main__":
    # IMU simulée : 6 canaux
    imu_raw = pd.DataFrame(
        np.random.randn(10000, 6),
        columns=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    )

    original_freq = 30  # ex: Ego4D / MMEA

    preproc = IMUVideoPreprocessing()

    imu_clean = preproc.preprocess_imu(imu_raw, original_freq)

    # Simulation frames vidéo alignées (indices)
    video_frames = np.arange(len(imu_clean))

    X_imu, X_video = preproc.extract_windows(imu_clean, video_frames)

    print("IMU windows shape :", X_imu.shape)      # [N, 250, 6]
    print("Video windows shape :", X_video.shape)  # [N, 10]
