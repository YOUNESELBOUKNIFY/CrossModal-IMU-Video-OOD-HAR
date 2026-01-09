# datasets.py
# PyTorch datasets:
# - CrossModalDataset: retourne (imu_window, video_clip_or_features, label)
# - IMUClassificationDataset: retourne (imu_window, label)
#
# Le dataset s’appuie sur un index généré par preprocessing.py (csv/parquet/jsonl)
#
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


# ---------------------------
# Index loading
# ---------------------------

def load_index(index_path: str) -> pd.DataFrame:
    ext = os.path.splitext(index_path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(index_path)
    if ext == ".csv":
        return pd.read_csv(index_path)
    if ext == ".jsonl":
        rows = []
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported index extension: {ext}")


# ---------------------------
# IMU loading
# ---------------------------

def load_imu_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr
    if ext == ".npz":
        z = np.load(path)
        key = list(z.keys())[0]
        arr = z[key]
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr
    if ext == ".csv":
        df = pd.read_csv(path)
        arr = df.values
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr
    if ext == ".txt":
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr
    raise ValueError(f"Unsupported IMU extension: {ext}")


# ---------------------------
# Video helpers (optionnels)
# ---------------------------

def _safe_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None

def read_video_clip_cv2(
    video_path: str,
    t_start: float,
    t_end: float,
    num_frames: int = 8,
    resize_hw: Optional[Tuple[int, int]] = (224, 224),
) -> np.ndarray:
    """
    Lit un clip vidéo entre [t_start, t_end] et échantillonne num_frames frames.
    Retour: np.ndarray shape (num_frames, H, W, 3) uint8

    ⚠️ Lent sur Kaggle si utilisé brut. Pour accélérer:
    - pré-extraction features vidéo (npys)
    - ou extraction frames au preprocessing
    """
    cv2 = _safe_import_cv2()
    if cv2 is None:
        raise ImportError("cv2 not available. Install opencv-python or use video_features instead.")

    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    start_frame = int(round(t_start * fps))
    end_frame = max(start_frame + 1, int(round(t_end * fps)))

    # échantillonnage linéaire de frames dans [start_frame, end_frame)
    frame_ids = np.linspace(start_frame, end_frame - 1, num_frames).round().astype(int)
    frames = []

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, frame = cap.read()
        if not ok:
            # fallback: frame noire si fin vidéo
            if resize_hw is None:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = np.zeros((resize_hw[0], resize_hw[1], 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize_hw is not None:
                frame = cv2.resize(frame, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA)

        frames.append(frame)

    cap.release()
    clip = np.stack(frames, axis=0).astype(np.uint8)
    return clip


# ---------------------------
# Datasets
# ---------------------------

class CrossModalDataset(Dataset):
    """
    Dataset cross-modal: (IMU window, VIDEO clip OR VIDEO features, label)

    Modes:
    1) raw video clip via cv2 (slow):
       - set use_video_clip=True and video_features_dir=None
    2) features pré-extraites:
       - set video_features_dir="/path/to/features"
       - feature file name must match sample_id (default) or can be customized.

    Expected index columns:
    - split, sample_id, label, imu_path, imu_start, imu_end, t_start, t_end, video_path(optional)
    """
    def __init__(
        self,
        index_path: str,
        split: str,
        imu_normalize: bool = True,
        imu_mean: Optional[np.ndarray] = None,
        imu_std: Optional[np.ndarray] = None,
        use_video_clip: bool = False,
        video_num_frames: int = 8,
        video_resize_hw: Optional[Tuple[int, int]] = (224, 224),
        video_features_dir: Optional[str] = None,
        video_feature_ext: str = ".npy",
        return_sample_id: bool = False,
    ):
        self.df = load_index(index_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split='{split}' in {index_path}")

        self.split = split
        self.imu_normalize = imu_normalize
        self.imu_mean = imu_mean
        self.imu_std = imu_std

        self.use_video_clip = use_video_clip
        self.video_num_frames = video_num_frames
        self.video_resize_hw = video_resize_hw

        self.video_features_dir = video_features_dir
        self.video_feature_ext = video_feature_ext

        if self.video_features_dir is None and not self.use_video_clip:
            # par défaut: si pas de features, on n’utilise pas vidéo => erreur (car cross-modal)
            raise ValueError("CrossModalDataset requires either use_video_clip=True or video_features_dir!=None")

        self.return_sample_id = return_sample_id

    def __len__(self) -> int:
        return len(self.df)

    def _load_imu_window(self, imu_path: str, s: int, e: int) -> torch.Tensor:
        arr = load_imu_array(imu_path)  # (T, C)
        window = arr[s:e]               # (W, C)
        window = window.astype(np.float32)

        if self.imu_normalize:
            if self.imu_mean is not None and self.imu_std is not None:
                window = (window - self.imu_mean) / (self.imu_std + 1e-8)
            else:
                # per-window normalization (simple baseline)
                m = window.mean(axis=0, keepdims=True)
                sd = window.std(axis=0, keepdims=True)
                window = (window - m) / (sd + 1e-8)

        return torch.from_numpy(window)  # (W, C)

    def _load_video(self, row: pd.Series) -> torch.Tensor:
        sample_id = str(row["sample_id"])

        # 1) features pré-extraites
        if self.video_features_dir is not None:
            feat_path = os.path.join(self.video_features_dir, sample_id + self.video_feature_ext)
            if not os.path.exists(feat_path):
                raise FileNotFoundError(
                    f"Video feature not found: {feat_path}\n"
                    f"Tip: ensure features are saved as {sample_id}{self.video_feature_ext}"
                )
            feat = np.load(feat_path).astype(np.float32)
            return torch.from_numpy(feat)

        # 2) clip brut via cv2
        video_path = row.get("video_path", None)
        if video_path is None or (isinstance(video_path, float) and np.isnan(video_path)):
            raise FileNotFoundError("video_path is missing in index row. Provide video features or valid video_path.")

        clip = read_video_clip_cv2(
            video_path=str(video_path),
            t_start=float(row["t_start"]),
            t_end=float(row["t_end"]),
            num_frames=self.video_num_frames,
            resize_hw=self.video_resize_hw,
        )  # (F, H, W, 3) uint8

        # to float tensor (F, 3, H, W) in [0,1]
        clip = clip.astype(np.float32) / 255.0
        clip = np.transpose(clip, (0, 3, 1, 2))
        return torch.from_numpy(clip)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imu = self._load_imu_window(
            imu_path=str(row["imu_path"]),
            s=int(row["imu_start"]),
            e=int(row["imu_end"]),
        )
        video = self._load_video(row)
        label = int(row["label"])

        if self.return_sample_id:
            return imu, video, label, str(row["sample_id"])
        return imu, video, label


class IMUClassificationDataset(Dataset):
    """
    Dataset classification IMU-only: (imu_window, label)

    Expected index columns:
    - split, label, imu_path, imu_start, imu_end
    """
    def __init__(
        self,
        index_path: str,
        split: str,
        imu_normalize: bool = True,
        imu_mean: Optional[np.ndarray] = None,
        imu_std: Optional[np.ndarray] = None,
        return_sample_id: bool = False,
    ):
        self.df = load_index(index_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split='{split}' in {index_path}")

        self.split = split
        self.imu_normalize = imu_normalize
        self.imu_mean = imu_mean
        self.imu_std = imu_std
        self.return_sample_id = return_sample_id

    def __len__(self) -> int:
        return len(self.df)

    def _load_imu_window(self, imu_path: str, s: int, e: int) -> torch.Tensor:
        arr = load_imu_array(imu_path)
        window = arr[s:e].astype(np.float32)

        if self.imu_normalize:
            if self.imu_mean is not None and self.imu_std is not None:
                window = (window - self.imu_mean) / (self.imu_std + 1e-8)
            else:
                m = window.mean(axis=0, keepdims=True)
                sd = window.std(axis=0, keepdims=True)
                window = (window - m) / (sd + 1e-8)

        return torch.from_numpy(window)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imu = self._load_imu_window(
            imu_path=str(row["imu_path"]),
            s=int(row["imu_start"]),
            e=int(row["imu_end"]),
        )
        label = int(row["label"])

        if self.return_sample_id:
            return imu, label, str(row["sample_id"])
        return imu, label


# ---------------------------
# Optional: compute global mean/std (train split)
# ---------------------------

def compute_imu_mean_std(
    index_path: str,
    split: str = "train",
    max_windows: Optional[int] = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule mean/std global sur des fenêtres IMU du split.
    Utile si tu veux une normalisation stable (meilleur que per-window).
    """
    df = load_index(index_path)
    df = df[df["split"] == split].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No rows for split='{split}'")

    if max_windows is not None and len(df) > max_windows:
        df = df.sample(max_windows, random_state=0).reset_index(drop=True)

    sums = None
    sums2 = None
    count = 0

    for _, row in df.iterrows():
        arr = load_imu_array(str(row["imu_path"]))
        s = int(row["imu_start"])
        e = int(row["imu_end"])
        w = arr[s:e].astype(np.float64)  # (W, C)
        if w.ndim == 1:
            w = w[:, None]

        if sums is None:
            sums = w.sum(axis=0)
            sums2 = (w ** 2).sum(axis=0)
        else:
            sums += w.sum(axis=0)
            sums2 += (w ** 2).sum(axis=0)

        count += w.shape[0]

    mean = (sums / count).astype(np.float32)
    var = (sums2 / count - mean.astype(np.float64) ** 2).astype(np.float32)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    return mean, std
