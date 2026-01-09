# preprocessing.py
# Preprocessing du dataset UESTC-MMEA-CL (MMEA)
#
# Objectif:
# - Lire train/val/test.txt
# - Construire un index de fenêtres IMU synchronisées avec des clips vidéo
# - Sauvegarder un fichier d'index (parquet/csv) pour accélérer l'entraînement
#
# Usage (Kaggle):
# python preprocessing.py \
#   --root /kaggle/input/dataset-har/UESTC-MMEA-CL \
#   --out_dir /kaggle/working/mmea_index \
#   --window_sec 2.0 --stride_sec 0.5 --imu_rate 50 \
#   --save_format parquet
#
from __future__ import annotations

import os
import re
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Helpers: parsing split files
# ---------------------------

def _try_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def _try_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def parse_split_line(line: str) -> Dict[str, Any]:
    """
    Parse une ligne de train/val/test.txt.

    Formats possibles (exemples):
    - "sensor/xxx.npy video/xxx.mp4 12"
    - "xxx.npy xxx.mp4 12"
    - "sensor_path label" (sans video)
    - "id label ..." (autres variantes)

    Retour standard:
    {
      "imu_rel": str (chemin relatif capteur) ou None,
      "video_rel": str (chemin relatif video) ou None,
      "label": int ou None,
      "meta": dict (autres champs)
    }
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return {"imu_rel": None, "video_rel": None, "label": None, "meta": {}}

    # split par espaces ou virgules
    parts = re.split(r"[,\s]+", line)
    parts = [p for p in parts if p]

    imu_rel = None
    video_rel = None
    label = None
    meta: Dict[str, Any] = {}

    # Heuristiques: chercher un token avec extension capteur / video
    for p in parts:
        pl = p.lower()
        if pl.endswith((".npy", ".npz", ".csv", ".txt")) and ("sensor" in pl or "imu" in pl):
            imu_rel = p
        elif pl.endswith((".npy", ".npz")) and imu_rel is None:
            # parfois le split contient directement un fichier numpy capteur sans "sensor"
            imu_rel = p
        if pl.endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_rel = p

    # label: dernier entier plausible
    for p in reversed(parts):
        v = _try_int(p)
        if v is not None:
            label = v
            break

    # Si le format est "imu video label" sans mot "sensor"/"video", on tente:
    if imu_rel is None and len(parts) >= 2:
        # si la 1ère colonne ressemble à un fichier
        if parts[0].lower().endswith((".npy", ".npz", ".csv", ".txt")):
            imu_rel = parts[0]
        # si la 2ème colonne ressemble à un fichier video
        if parts[1].lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_rel = parts[1]

    # meta: garder la ligne brute + tokens
    meta["raw"] = line
    meta["tokens"] = parts

    return {"imu_rel": imu_rel, "video_rel": video_rel, "label": label, "meta": meta}


# ---------------------------
# IMU loading + windowing
# ---------------------------

def load_imu_array(path: str) -> np.ndarray:
    """
    Charge un fichier capteur.
    Supporte .npy/.npz/.csv (basique).
    Retour: np.ndarray shape (T, C)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        if isinstance(arr, np.ndarray):
            return arr
        raise ValueError(f"Unsupported npy object type: {type(arr)}")
    if ext == ".npz":
        z = np.load(path)
        # prendre le premier array
        key = list(z.keys())[0]
        return z[key]
    if ext == ".csv":
        df = pd.read_csv(path)
        return df.values
    if ext == ".txt":
        # txt numérique: whitespace separated
        arr = np.loadtxt(path)
        return arr
    raise ValueError(f"Unsupported IMU file extension: {ext}")

def compute_windows(total_len: int, window_size: int, stride: int) -> List[Tuple[int, int]]:
    """
    Retourne liste de (start, end) inclusif-exclusif pour des fenêtres.
    """
    if total_len < window_size:
        return []
    windows = []
    start = 0
    while start + window_size <= total_len:
        end = start + window_size
        windows.append((start, end))
        start += stride
    return windows


# ---------------------------
# Index schema
# ---------------------------

@dataclass
class IndexRow:
    split: str
    sample_id: str
    label: int
    imu_path: str
    video_path: Optional[str]
    # fenêtre IMU
    imu_start: int
    imu_end: int
    # optionnel: timestamps (secondes) si on veut faire correspondre à la vidéo
    t_start: float
    t_end: float


def make_sample_id(split: str, i: int, imu_rel: Optional[str], video_rel: Optional[str]) -> str:
    base = f"{split}_{i:06d}"
    if imu_rel:
        base += "_" + os.path.splitext(os.path.basename(imu_rel))[0]
    if video_rel:
        base += "_" + os.path.splitext(os.path.basename(video_rel))[0]
    # nettoyer
    base = re.sub(r"[^a-zA-Z0-9_\-]+", "_", base)
    return base


# ---------------------------
# Main preprocessing pipeline
# ---------------------------

def build_index_for_split(
    split_name: str,
    split_txt_path: str,
    root: str,
    imu_dirname: str = "sensor",
    video_dirname: str = "video",
    imu_rate: int = 50,
    window_sec: float = 2.0,
    stride_sec: float = 0.5,
    max_windows_per_sample: Optional[int] = None,
) -> pd.DataFrame:
    """
    Construit un DataFrame index pour un split.
    """
    window_size = int(round(window_sec * imu_rate))
    stride = int(round(stride_sec * imu_rate))
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_sec/stride_sec invalid (<=0)")

    rows: List[Dict[str, Any]] = []

    with open(split_txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]

    for i, line in enumerate(lines):
        parsed = parse_split_line(line)
        imu_rel = parsed["imu_rel"]
        video_rel = parsed["video_rel"]
        label = parsed["label"]

        if label is None:
            raise ValueError(f"[{split_name}] Label introuvable à la ligne {i}: {line}")

        # Construire chemins absolus
        imu_path = None
        if imu_rel:
            # si le split donne déjà "sensor/xxx", on join root direct
            imu_path = os.path.join(root, imu_rel)
            if not os.path.exists(imu_path):
                # sinon on tente root/sensor/imu_rel
                imu_path2 = os.path.join(root, imu_dirname, imu_rel)
                if os.path.exists(imu_path2):
                    imu_path = imu_path2
        else:
            # Si le format n’inclut pas le chemin capteur, impossible
            raise ValueError(f"[{split_name}] imu_rel introuvable à la ligne {i}: {line}")

        if imu_path is None or not os.path.exists(imu_path):
            raise FileNotFoundError(f"IMU file not found for line {i}: {line}")

        video_path = None
        if video_rel:
            video_path = os.path.join(root, video_rel)
            if not os.path.exists(video_path):
                video_path2 = os.path.join(root, video_dirname, video_rel)
                if os.path.exists(video_path2):
                    video_path = video_path2
                else:
                    # on accepte video manquante (tu peux faire IMU-only)
                    video_path = None

        # Charger IMU pour connaître T
        imu_arr = load_imu_array(imu_path)
        if imu_arr.ndim == 1:
            imu_arr = imu_arr[:, None]
        T = int(imu_arr.shape[0])

        windows = compute_windows(T, window_size, stride)
        if not windows:
            continue

        if max_windows_per_sample is not None and len(windows) > max_windows_per_sample:
            # sous-échantillonner uniformément
            idxs = np.linspace(0, len(windows) - 1, max_windows_per_sample).round().astype(int)
            windows = [windows[j] for j in idxs]

        sample_id_base = make_sample_id(split_name, i, imu_rel, video_rel)

        for wj, (s, e) in enumerate(windows):
            t_start = s / float(imu_rate)
            t_end = e / float(imu_rate)

            row = IndexRow(
                split=split_name,
                sample_id=f"{sample_id_base}_w{wj:04d}",
                label=int(label),
                imu_path=os.path.abspath(imu_path),
                video_path=os.path.abspath(video_path) if video_path else None,
                imu_start=int(s),
                imu_end=int(e),
                t_start=float(t_start),
                t_end=float(t_end),
            )
            rows.append(asdict(row))

    return pd.DataFrame(rows)


def save_index(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".parquet":
        df.to_parquet(out_path, index=False)
    elif ext == ".csv":
        df.to_csv(out_path, index=False)
    elif ext == ".jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported save format: {ext}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Racine du dataset (ex: /kaggle/input/.../UESTC-MMEA-CL)")
    ap.add_argument("--out_dir", type=str, required=True, help="Dossier de sortie pour l'index")
    ap.add_argument("--imu_dirname", type=str, default="sensor")
    ap.add_argument("--video_dirname", type=str, default="video")

    ap.add_argument("--train_txt", type=str, default="train.txt")
    ap.add_argument("--val_txt", type=str, default="val.txt")
    ap.add_argument("--test_txt", type=str, default="test.txt")

    ap.add_argument("--imu_rate", type=int, default=50)
    ap.add_argument("--window_sec", type=float, default=2.0)
    ap.add_argument("--stride_sec", type=float, default=0.5)
    ap.add_argument("--max_windows_per_sample", type=int, default=None)

    ap.add_argument("--save_format", type=str, default="parquet", choices=["parquet", "csv", "jsonl"])

    args = ap.parse_args()

    root = args.root
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    split_files = {
        "train": os.path.join(root, args.train_txt),
        "val": os.path.join(root, args.val_txt),
        "test": os.path.join(root, args.test_txt),
    }

    all_dfs = []
    for split_name, split_path in split_files.items():
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")

        df = build_index_for_split(
            split_name=split_name,
            split_txt_path=split_path,
            root=root,
            imu_dirname=args.imu_dirname,
            video_dirname=args.video_dirname,
            imu_rate=args.imu_rate,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            max_windows_per_sample=args.max_windows_per_sample,
        )
        print(f"{split_name}: {len(df)} windows")
        all_dfs.append(df)

    full = pd.concat(all_dfs, ignore_index=True)

    out_path = os.path.join(out_dir, f"mmea_index.{args.save_format}")
    save_index(full, out_path)
    print(f"Saved index to: {out_path}")
    print(full.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
