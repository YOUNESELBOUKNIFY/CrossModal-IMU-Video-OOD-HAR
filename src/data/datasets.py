"""
PyTorch Datasets pour IMU-Video Cross-modal Learning
Conforme à l'article avec gestion robuste des vidéos manquantes

✅ Fix importants (par rapport à ton ancien code):
- Résolution robuste des chemins imu_window_path (absolu/relatif) + fichier manquant
- Vidéo: gère fps=0, total_frames=0, start_frame hors bornes, frames noires
- Option pour retourner vidéo en (T,C,H,W) ou (C,T,H,W) via config.data.video_channel_first
- create_dataloaders: pin_memory, drop_last, persistent_workers (si num_workers>0)
- Quelques prints propres + validations
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# -----------------------------
#   Cross-modal dataset
# -----------------------------
class CrossModalDataset(Dataset):
    """
    Dataset pour l'entraînement cross-modal IMU-Video
    Retourne: {'imu': (C,T), 'video': (T,C,H,W) ou (C,T,H,W), 'idx': int}
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        config,
        video_transform=None,
        return_paths: bool = False,
    ):
        self.df = metadata_df.reset_index(drop=True).copy()
        self.config = config
        self.data_cfg = config.data
        self.paths_cfg = config.paths
        self.return_paths = return_paths

        # ----- Video transforms -----
        if video_transform is None:
            self.video_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.data_cfg.video_resize),  # (H, W)
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.video_transform = video_transform

        # Info
        print(f"[CrossModalDataset] {len(self.df)} samples")
        if "video_exists" in self.df.columns:
            n_with = int(self.df["video_exists"].sum())
            print(f"  - with video: {n_with}")
            print(f"  - without video: {len(self.df) - n_with}")

        # Option: format vidéo pour ton modèle
        # False => (T,C,H,W) (par défaut)
        # True  => (C,T,H,W)
        self.video_channel_first = bool(getattr(self.data_cfg, "video_channel_first", False))

    def __len__(self):
        return len(self.df)

    def _resolve_imu_path(self, imu_path: str) -> Path:
        """
        imu_window_path peut être:
        - absolu (/kaggle/working/...)
        - relatif (preprocessed/train/xxx.npy, ou train/xxx.npy, etc.)
        """
        p = Path(str(imu_path))

        # Si déjà absolu -> ok
        if p.is_absolute():
            return p

        # Sinon: essayer par rapport à preprocessed_dir
        # 1) direct
        cand1 = self.paths_cfg.preprocessed_dir / p
        if cand1.exists():
            return cand1

        # 2) si on a juste "train/xxx.npy" (ok déjà)
        # 3) fallback: tenter /kaggle/working + p
        cand2 = Path.cwd() / p
        if cand2.exists():
            return cand2

        # sinon retourner cand1 (ça va déclencher erreur gérée)
        return cand1

    def load_imu_window(self, imu_path: str) -> torch.Tensor:
        """
        Charge un window IMU prétraité
        Returns: tensor shape (C, T) pour Conv1D
        """
        try:
            imu_file = self._resolve_imu_path(imu_path)

            if not imu_file.exists():
                # fichier manquant
                return torch.zeros(self.data_cfg.imu_channels, self.data_cfg.imu_window_size)

            imu_data = np.load(str(imu_file))  # (T, C) attendu
            imu_data = np.asarray(imu_data, dtype=np.float32)

            # Sécuriser shape
            T = self.data_cfg.imu_window_size
            C = self.data_cfg.imu_channels

            if imu_data.ndim != 2:
                return torch.zeros(C, T)

            # Si dimensions inversées (C,T) -> remettre (T,C)
            if imu_data.shape == (C, T):
                imu_data = imu_data.T

            # Si pas la bonne taille -> pad/crop
            if imu_data.shape[0] != T or imu_data.shape[1] != C:
                out = np.zeros((T, C), dtype=np.float32)
                t_min = min(T, imu_data.shape[0])
                c_min = min(C, imu_data.shape[1])
                out[:t_min, :c_min] = imu_data[:t_min, :c_min]
                imu_data = out

            # (T,C) -> (C,T)
            imu_tensor = torch.from_numpy(imu_data).transpose(0, 1)
            return imu_tensor

        except Exception as e:
            print(f"[CrossModalDataset] IMU load error: {imu_path} -> {e}")
            return torch.zeros(self.data_cfg.imu_channels, self.data_cfg.imu_window_size)

    def _black_video(self) -> torch.Tensor:
        num_frames = int(self.data_cfg.video_frames_per_window)
        H, W = self.data_cfg.video_resize
        video = torch.zeros(num_frames, 3, H, W)
        if self.video_channel_first:
            video = video.permute(1, 0, 2, 3)  # (C,T,H,W)
        return video

    def load_video_clip(self, video_path: str, start_frame: int) -> torch.Tensor:
        """
        Charge un clip vidéo de N frames
        - extrait video_frames_per_window frames uniformément sur 5 secondes
        Returns:
          - (T,C,H,W) ou (C,T,H,W)
        """
        full_video_path = self.paths_cfg.base_input / str(video_path)

        if not full_video_path.exists():
            return self._black_video()

        try:
            cap = cv2.VideoCapture(str(full_video_path))
            if not cap.isOpened():
                cap.release()
                return self._black_video()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0

            if total_frames <= 0:
                cap.release()
                return self._black_video()

            # Si fps inconnu -> fallback: considérer 25
            if fps <= 1e-6:
                fps = float(getattr(self.data_cfg, "video_fps", 25.0))

            # Fenêtre IMU = 5s -> fenêtre vidéo = 5s * fps
            window_sec = self.data_cfg.imu_window_size / float(self.data_cfg.imu_sampling_rate)
            window_frames = max(int(round(window_sec * fps)), 1)

            target_frames = int(self.data_cfg.video_frames_per_window)

            # start_frame robuste
            start_frame = int(start_frame)
            if start_frame < 0:
                start_frame = 0
            if start_frame >= total_frames:
                start_frame = max(total_frames - 1, 0)

            end_frame = start_frame + window_frames - 1
            end_frame = min(end_frame, total_frames - 1)

            # Indices uniformes
            if end_frame >= start_frame:
                frame_indices = np.linspace(start_frame, end_frame, target_frames, dtype=int)
            else:
                frame_indices = np.full((target_frames,), start_frame, dtype=int)

            frame_indices = np.clip(frame_indices, 0, total_frames - 1)

            H, W = self.data_cfg.video_resize
            frames = []

            for fi in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ret, frame = cap.read()

                if not ret or frame is None:
                    frames.append(torch.zeros(3, H, W))
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = self.video_transform(frame_rgb)  # (C,H,W)
                frames.append(frame_tensor)

            cap.release()

            # Stack => (T,C,H,W)
            video_tensor = torch.stack(frames, dim=0)

            if self.video_channel_first:
                video_tensor = video_tensor.permute(1, 0, 2, 3)  # (C,T,H,W)

            return video_tensor

        except Exception as e:
            print(f"[CrossModalDataset] Video load error: {video_path} -> {e}")
            return self._black_video()

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        imu_tensor = self.load_imu_window(row["imu_window_path"])
        start_frame = int(row["start_frame"]) if "start_frame" in row else 0
        video_tensor = self.load_video_clip(row["video_path"], start_frame)

        out = {"imu": imu_tensor, "video": video_tensor, "idx": idx}

        if self.return_paths:
            out["imu_path"] = row["imu_window_path"]
            out["video_path"] = row["video_path"]

        return out


# -----------------------------
#   IMU-only classification dataset
# -----------------------------
class IMUClassificationDataset(Dataset):
    """
    Dataset pour classification IMU seul
    Retourne: {'imu': (C,T), 'label': int, 'idx': int}
    """

    def __init__(self, metadata_df: pd.DataFrame, config, return_info: bool = False):
        self.df = metadata_df.reset_index(drop=True).copy()
        self.config = config
        self.data_cfg = config.data
        self.paths_cfg = config.paths
        self.return_info = return_info

        print(f"[IMUClassificationDataset] {len(self.df)} samples")
        if "label" in self.df.columns:
            print(f"  - classes: {self.df['label'].nunique()}")

    def __len__(self):
        return len(self.df)

    def _resolve_imu_path(self, imu_path: str) -> Path:
        p = Path(str(imu_path))
        if p.is_absolute():
            return p

        cand1 = self.paths_cfg.preprocessed_dir / p
        if cand1.exists():
            return cand1

        cand2 = Path.cwd() / p
        if cand2.exists():
            return cand2

        return cand1

    def load_imu_window(self, imu_path: str) -> torch.Tensor:
        try:
            imu_file = self._resolve_imu_path(imu_path)
            if not imu_file.exists():
                return torch.zeros(self.data_cfg.imu_channels, self.data_cfg.imu_window_size)

            imu_data = np.load(str(imu_file))
            imu_data = np.asarray(imu_data, dtype=np.float32)

            T = self.data_cfg.imu_window_size
            C = self.data_cfg.imu_channels

            if imu_data.ndim != 2:
                return torch.zeros(C, T)

            if imu_data.shape == (C, T):
                imu_data = imu_data.T

            if imu_data.shape[0] != T or imu_data.shape[1] != C:
                out = np.zeros((T, C), dtype=np.float32)
                t_min = min(T, imu_data.shape[0])
                c_min = min(C, imu_data.shape[1])
                out[:t_min, :c_min] = imu_data[:t_min, :c_min]
                imu_data = out

            imu_tensor = torch.from_numpy(imu_data).transpose(0, 1)  # (C,T)
            return imu_tensor

        except Exception as e:
            print(f"[IMUClassificationDataset] IMU load error: {imu_path} -> {e}")
            return torch.zeros(self.data_cfg.imu_channels, self.data_cfg.imu_window_size)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        imu_tensor = self.load_imu_window(row["imu_window_path"])
        label = int(row["label"])

        out = {"imu": imu_tensor, "label": label, "idx": idx}

        if self.return_info:
            if "class_name" in row:
                out["class_name"] = row["class_name"]
            if "user_id" in row:
                out["user_id"] = row["user_id"]

        return out


# -----------------------------
#   Few-shot sampler
# -----------------------------
class FewShotSampler:
    def __init__(self, metadata_df: pd.DataFrame, config):
        self.df = metadata_df
        self.config = config

    def sample_k_per_class(self, k: int, seed: Optional[int] = None) -> pd.DataFrame:
        if seed is not None:
            np.random.seed(seed)

        sampled = []
        for class_name in self.df["class_name"].unique():
            class_df = self.df[self.df["class_name"] == class_name]
            if len(class_df) >= k:
                subset = class_df.sample(n=k, random_state=seed)
            else:
                subset = class_df
            sampled.append(subset)

        result_df = pd.concat(sampled, ignore_index=True)
        print(f"[FewShotSampler] {len(result_df)} samples ({k}/class × {self.df['class_name'].nunique()} classes)")
        return result_df

    def sample_balanced_test_set(self, n_per_class: int = 20, seed: Optional[int] = None) -> pd.DataFrame:
        return self.sample_k_per_class(n_per_class, seed)


# -----------------------------
#   Dataloaders factory
# -----------------------------
def create_dataloaders(
    config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str = "cross_modal",
    shuffle_train: bool = True
) -> Dict[str, DataLoader]:

    if mode == "cross_modal":
        print("[create_dataloaders] cross-modal")
        train_dataset = CrossModalDataset(train_df, config)
        val_dataset = CrossModalDataset(val_df, config)
        test_dataset = CrossModalDataset(test_df, config)
        batch_size = config.training.pretrain_batch_size

    elif mode == "classification":
        print("[create_dataloaders] classification")
        train_dataset = IMUClassificationDataset(train_df, config)
        val_dataset = IMUClassificationDataset(val_df, config)
        test_dataset = IMUClassificationDataset(test_df, config)
        batch_size = config.training.train_batch_size

    else:
        raise ValueError(f"Unknown mode: {mode}")

    num_workers = int(config.training.num_workers)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    print("✓ Dataloaders:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val  : {len(val_loader)} batches")
    print(f"  Test : {len(test_loader)} batches")

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def get_class_weights(metadata_df: pd.DataFrame) -> torch.Tensor:
    class_counts = metadata_df["label"].value_counts().sort_index()
    total = len(metadata_df)
    weights = total / (len(class_counts) * class_counts.values)
    return torch.FloatTensor(weights)


# -----------------------------
#   Quick test
# -----------------------------
def test_dataloaders():
    import sys
    sys.path.append(".")
    from config import CONFIG

    print("\n" + "=" * 60)
    print("TEST DATALOADERS")
    print("=" * 60)

    train_df = pd.read_csv(CONFIG.paths.preprocessed_dir / "train_metadata.csv")
    print(f"Loaded train_df: {len(train_df)}")

    ds = CrossModalDataset(train_df.head(3), CONFIG, return_paths=True)
    s = ds[0]
    print("CrossModal sample shapes:")
    print("  imu:", s["imu"].shape)      # (C,T)
    print("  vid:", s["video"].shape)   # (T,C,H,W) or (C,T,H,W)

    ds2 = IMUClassificationDataset(train_df.head(3), CONFIG, return_info=True)
    s2 = ds2[0]
    print("IMUCls sample shapes:")
    print("  imu:", s2["imu"].shape, "label:", s2["label"])

    print("✓ OK")


if __name__ == "__main__":
    test_dataloaders()
