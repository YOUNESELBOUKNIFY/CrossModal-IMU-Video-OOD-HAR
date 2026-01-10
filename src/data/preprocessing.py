"""
Preprocessing complet du dataset MMEA pour IMU-Video Cross-modal Learning

Format split (train/val/test.txt) observé :
  yourdataset_path/data/<class_dir>/<sample_prefix> <start> <end> <label>

Note dataset :
- Video: ./video/[class_number]_[class_name]/xx.mp4
- Sensor: ./sensor/[class_number]_[class_name]/xx.csv
- Même prefix = même sample
- CSV: 6 colonnes = acc(x,y,z), gyro(x,y,z) en RAW
- Conversion: acc_g = raw_acc / 16384 ; gyro_deg_s = raw_gyro / 16.4

Preprocess:
- Fenêtres IMU de 5s à 50Hz (250)
- stride configurable (ex 125)
- median filter + z-score
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import scipy.signal as signal
import warnings
warnings.filterwarnings("ignore")


class MMEAPreprocessor:
    """
    Préprocesseur pour dataset MMEA
    - Parse splits (format data/.. + start/end/label)
    - Charge IMU (.csv) + conversion Racc/Rgyro
    - Crée fenêtres IMU 5s@50Hz
    - Associe la vidéo correspondante (.mp4)
    """

    def __init__(self, config):
        self.config = config
        self.paths = config.paths
        self.data_cfg = config.data

        self.preprocessing_stats = {
            "total_samples": 0,
            "skipped_samples": 0,
            "total_windows": 0,
            "samples_with_video": 0,
            "samples_without_video": 0,
            "classes_found": set(),
            "bad_format_lines": 0,
            "missing_sensor_files": 0,
            "missing_video_files": 0,
            "prefix_mismatch": 0,
            "too_short_no_pad": 0,
        }

    # -------------------------
    # Split loading
    # -------------------------
    def load_split(self, split: str) -> List[str]:
        if split == "train":
            split_file = self.paths.base_input / self.paths.train_file
        elif split == "val":
            split_file = self.paths.base_input / self.paths.val_file
        elif split == "test":
            split_file = self.paths.base_input / self.paths.test_file
        else:
            raise ValueError(f"Split inconnu: {split}")

        if not split_file.exists():
            raise FileNotFoundError(f"Fichier split introuvable: {split_file}")

        lines = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    lines.append(line)

        print(f"[{split}] Chargé {len(lines)} lignes depuis {split_file}")
        return lines

    # -------------------------
    # Parse split line -> sensor/video paths
    # -------------------------
    def parse_split_line(self, line: str) -> Dict:
        """
        line ex:
          yourdataset_path/data/27_open_close_door/27_open_close_door_2022_05_05_16_05_51 418 419 26

        Returns:
          class_dir, class_num, class_name, sample_prefix, sensor_path, video_path, start, end, label
        """
        toks = line.strip().split()
        if len(toks) < 4:
            self.preprocessing_stats["bad_format_lines"] += 1
            raise ValueError(f"Ligne split invalide: {line}")

        path_token = toks[0]
        start = int(toks[1])
        end = int(toks[2])
        label = int(toks[3])

        # sécurité start/end
        if start > end:
            start, end = end, start

        p = Path(path_token)
        parts = p.parts

        # On s'attend à .../data/<class_dir>/<sample_prefix>
        if "data" not in parts:
            self.preprocessing_stats["bad_format_lines"] += 1
            raise ValueError(f"Pas de dossier 'data' dans: {line}")

        i = parts.index("data")
        if len(parts) < i + 3:
            self.preprocessing_stats["bad_format_lines"] += 1
            raise ValueError(f"Format .../data/<class>/<sample> invalide: {line}")

        class_dir = parts[i + 1]        # "27_open_close_door"
        sample_prefix = parts[i + 2]    # "27_open_close_door_2022_..."

        # Split class_dir => (num, name)
        if "_" in class_dir:
            class_num_str, class_name = class_dir.split("_", 1)
            class_num = int(class_num_str) if class_num_str.isdigit() else -1
        else:
            class_num = -1
            class_name = class_dir

        # Construire chemins
        sensor_rel = str(Path("sensor") / class_dir / f"{sample_prefix}.csv")
        video_rel = str(Path("video") / class_dir / f"{sample_prefix}.mp4")

        return {
            "class_dir": class_dir,
            "class_num": class_num,
            "class_name": class_name,
            "sample_prefix": sample_prefix,
            "sensor_path": sensor_rel,
            "video_path": video_rel,
            "start": start,
            "end": end,
            "label": label,
        }

    # -------------------------
    # IO helpers
    # -------------------------
    def exists(self, relpath: str) -> bool:
        return (self.paths.base_input / relpath).exists()

    def load_imu_data(self, sensor_relpath: str) -> Optional[np.ndarray]:
        """
        Charge CSV capteur (Nx6) + conversion Racc/Rgyro.
        """
        full_path = self.paths.base_input / sensor_relpath
        if not full_path.exists():
            self.preprocessing_stats["missing_sensor_files"] += 1
            return None

        try:
            data = pd.read_csv(full_path, header=None).values.astype(np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)

            # forcer 6 colonnes
            if data.shape[1] < 6:
                pad = np.zeros((data.shape[0], 6 - data.shape[1]), dtype=np.float32)
                data = np.hstack([data, pad])
            elif data.shape[1] > 6:
                data = data[:, :6]

            # conversion physique
            Racc = float(getattr(self.data_cfg, "Racc", 16384.0))
            Rgyro = float(getattr(self.data_cfg, "Rgyro", 16.4))

            acc = data[:, :3] / Racc
            gyro = data[:, 3:6] / Rgyro
            data = np.concatenate([acc, gyro], axis=1).astype(np.float32)
            return data

        except Exception:
            return None

    # -------------------------
    # Signal ops
    # -------------------------
    def resample_imu(self, imu: np.ndarray, original_rate: float, target_rate: float) -> np.ndarray:
        if original_rate == target_rate:
            return imu
        n = imu.shape[0]
        n_target = int(round(n * target_rate / original_rate))
        if n_target <= 1:
            return imu
        out = []
        for ch in range(imu.shape[1]):
            out.append(signal.resample(imu[:, ch], n_target))
        return np.stack(out, axis=1).astype(np.float32)

    def preprocess_imu(self, imu: np.ndarray) -> np.ndarray:
        # median filter
        k = int(getattr(self.data_cfg, "median_filter_kernel", 5))
        if k > 1:
            if k % 2 == 0:
                k += 1
            filtered = np.zeros_like(imu, dtype=np.float32)
            for ch in range(imu.shape[1]):
                filtered[:, ch] = signal.medfilt(imu[:, ch], kernel_size=k)
            imu = filtered

        # z-score par canal (par sample)
        if bool(getattr(self.data_cfg, "normalize_imu", True)):
            mean = imu.mean(axis=0, keepdims=True)
            std = imu.std(axis=0, keepdims=True) + 1e-8
            imu = (imu - mean) / std

        return imu.astype(np.float32)

    def create_imu_windows(self, imu: np.ndarray) -> List[np.ndarray]:
        windows = []
        window_size = int(getattr(self.data_cfg, "imu_window_size", 250))
        stride = int(getattr(self.data_cfg, "imu_stride", 125))

        n = imu.shape[0]

        if n < window_size:
            # pad ou skip
            if bool(getattr(self.data_cfg, "pad_short_sequences", True)):
                pad = np.zeros((window_size - n, imu.shape[1]), dtype=np.float32)
                imu = np.vstack([imu, pad])
                n = window_size
            else:
                self.preprocessing_stats["too_short_no_pad"] += 1
                return []

        for start in range(0, n - window_size + 1, stride):
            windows.append(imu[start:start + window_size])

        return windows

    def estimate_start_frame(self, window_idx: int) -> int:
        sr = float(getattr(self.data_cfg, "imu_sampling_rate", 50.0))
        stride = float(getattr(self.data_cfg, "imu_stride", 125))
        fps = float(getattr(self.data_cfg, "video_fps", 25.0))
        start_time = window_idx * (stride / sr)
        return int(round(start_time * fps))

    # -------------------------
    # Main split preprocessing
    # -------------------------
    def preprocess_split(self, split: str, save: bool = True) -> pd.DataFrame:
        lines = self.load_split(split)
        self.preprocessing_stats["total_samples"] += len(lines)

        records = []

        print(f"\n{'='*60}")
        print(f"Preprocessing du split: {split.upper()}")
        print(f"{'='*60}")

        # Option: cross-modal strict => skip samples sans vidéo
        require_video = bool(getattr(self.data_cfg, "require_video", False))

        # Option: resample si on connaît original_rate
        orig_rate = getattr(self.data_cfg, "imu_original_rate", None)
        target_rate = float(getattr(self.data_cfg, "imu_sampling_rate", 50.0))

        for line in tqdm(lines, desc=f"Processing {split}"):
            # 1) parse line
            try:
                info = self.parse_split_line(line)
            except Exception:
                self.preprocessing_stats["skipped_samples"] += 1
                continue

            self.preprocessing_stats["classes_found"].add(info["class_dir"])

            sensor_path = info["sensor_path"]
            video_path = info["video_path"]

            # 2) existence files
            sensor_ok = self.exists(sensor_path)
            video_ok = self.exists(video_path)

            if not sensor_ok:
                self.preprocessing_stats["missing_sensor_files"] += 1
                self.preprocessing_stats["skipped_samples"] += 1
                continue

            if video_ok:
                self.preprocessing_stats["samples_with_video"] += 1
            else:
                self.preprocessing_stats["samples_without_video"] += 1
                self.preprocessing_stats["missing_video_files"] += 1
                if require_video:
                    self.preprocessing_stats["skipped_samples"] += 1
                    continue

            # 3) prefix check
            if Path(sensor_path).stem != Path(video_path).stem:
                self.preprocessing_stats["prefix_mismatch"] += 1
                self.preprocessing_stats["skipped_samples"] += 1
                continue

            # 4) load imu
            imu_raw = self.load_imu_data(sensor_path)
            if imu_raw is None or imu_raw.size == 0:
                self.preprocessing_stats["skipped_samples"] += 1
                continue

            # 5) resample (si orig_rate est défini)
            if orig_rate is not None:
                imu_raw = self.resample_imu(imu_raw, float(orig_rate), target_rate)

            # 6) preprocess + windows
            imu_proc = self.preprocess_imu(imu_raw)
            windows = self.create_imu_windows(imu_proc)
            if not windows:
                self.preprocessing_stats["skipped_samples"] += 1
                continue

            # label : on utilise celui du split (plus fiable)
            label = info["label"]

            for w_idx, window in enumerate(windows):
                self.preprocessing_stats["total_windows"] += 1

                rec = {
                    "split": split,
                    "class_dir": info["class_dir"],
                    "class_name": info["class_name"],
                    "class_num": info["class_num"],
                    "label": label,
                    "sample_id": info["sample_prefix"],
                    "window_idx": w_idx,
                    "split_line": line,
                    "sensor_path": sensor_path,
                    "video_path": video_path,
                    "video_exists": video_ok,
                    "start_frame": self.estimate_start_frame(w_idx),
                    "imu_shape_0": int(window.shape[0]),
                    "imu_shape_1": int(window.shape[1]),
                    # optionnel : infos start/end du split (si tu veux les exploiter plus tard)
                    "split_start": info["start"],
                    "split_end": info["end"],
                }

                if save:
                    out_dir = self.paths.preprocessed_dir / split
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"{info['class_dir']}_{info['sample_prefix']}_w{w_idx}.npy"
                    fpath = out_dir / fname
                    np.save(fpath, window.astype(np.float32))
                    rec["imu_window_path"] = str(fpath)

                records.append(rec)

        df = pd.DataFrame(records)

        if save:
            self.paths.preprocessed_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self.paths.preprocessed_dir / f"{split}_metadata.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Metadata sauvegardée: {csv_path}")
            print(f"  Total windows: {len(df)}")
            if len(df) > 0:
                print(f"  Windows avec vidéo: {int(df['video_exists'].sum())}")
                print(f"  Windows sans vidéo: {int((~df['video_exists']).sum())}")

        return df

    def run_full_preprocessing(self):
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLET DU DATASET MMEA")
        print("=" * 60)

        results = {}
        for split in ["train", "val", "test"]:
            try:
                results[split] = self.preprocess_split(split, save=True)
            except FileNotFoundError:
                print(f"Avertissement: split '{split}' introuvable, ignoré")

        # stats
        print("\n" + "=" * 60)
        print("STATISTIQUES FINALES")
        print("=" * 60)
        print(f"Total échantillons (dans splits): {self.preprocessing_stats['total_samples']}")
        print(f"Échantillons skip: {self.preprocessing_stats['skipped_samples']}")
        print(f"Total windows créées: {self.preprocessing_stats['total_windows']}")
        print(f"Échantillons avec vidéo: {self.preprocessing_stats['samples_with_video']}")
        print(f"Échantillons sans vidéo: {self.preprocessing_stats['samples_without_video']}")
        print(f"Classes trouvées: {len(self.preprocessing_stats['classes_found'])}")
        print(f"Bad format lines: {self.preprocessing_stats['bad_format_lines']}")
        print(f"Missing sensor: {self.preprocessing_stats['missing_sensor_files']}")
        print(f"Missing video: {self.preprocessing_stats['missing_video_files']}")
        print(f"Prefix mismatch: {self.preprocessing_stats['prefix_mismatch']}")
        print(f"Too short (no pad): {self.preprocessing_stats['too_short_no_pad']}")

        stats_path = self.paths.preprocessed_dir / "preprocessing_stats.json"
        stats_to_save = dict(self.preprocessing_stats)
        stats_to_save["classes_found"] = sorted(list(stats_to_save["classes_found"]))
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_to_save, f, indent=2)
        print(f"\n✓ Statistiques sauvegardées: {stats_path}")

        print("\n" + "=" * 60)
        print("PREPROCESSING TERMINÉ")
        print("=" * 60)

        return results
