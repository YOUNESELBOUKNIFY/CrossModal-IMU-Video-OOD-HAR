"""
Preprocessing complet du dataset MMEA pour IMU-Video Cross-modal Learning

✅ Conforme à la note du dataset :
- Vidéo : ./video/[class_number]_[class_name]/xx.mp4
- Capteurs : ./sensor/[class_number]_[class_name]/xx.csv
- Même préfixe = même sample
- CSV : 6 colonnes (acc x,y,z ; gyro x,y,z) en valeurs brutes
- Conversion : acc_g = raw_acc / 16384, gyro_deg_s = raw_gyro / 16.4

✅ Fenêtrage (par défaut) :
- Fenêtres IMU 5s à 50Hz => 250 échantillons
- Stride configurable (par défaut 50% overlap => 125)
- Median filter + Z-score (optionnel)

⚠️ Important :
- On SUPPRIME les "fallback" dangereux : si un chemin est invalide => on skip.
- On construit le chemin vidéo à partir du chemin capteur, et on vérifie le préfixe.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.signal as signal
from tqdm import tqdm

warnings.filterwarnings("ignore")


# -----------------------------
# Config helpers (optionnels)
# -----------------------------
@dataclass
class DataConfig:
    imu_sampling_rate: float = 50.0          # cible (Hz)
    imu_original_rate: Optional[float] = None  # si None => pas de resample par défaut
    imu_window_size: int = 250               # 5s*50Hz
    imu_stride: int = 125                    # 50% overlap
    median_filter_kernel: int = 5
    normalize_imu: bool = True

    video_fps: float = 25.0                  # utilisé pour start_frame
    video_ext: str = ".mp4"

    # Sensitivity factors (dataset note)
    Racc: float = 16384.0
    Rgyro: float = 16.4

    # Si un sample est trop court, on peut soit pad, soit skip.
    pad_short_sequences: bool = True         # True = pad, False = skip
    min_windows_per_sample: int = 1          # si pad=False et trop court => 0 window => skip


@dataclass
class PathsConfig:
    base_input: Path
    preprocessed_dir: Path
    train_file: str = "train.txt"
    val_file: str = "val.txt"
    test_file: str = "test.txt"


@dataclass
class Config:
    paths: PathsConfig
    data: DataConfig


# -----------------------------
# Preprocessor
# -----------------------------
class MMEAPreprocessor:
    """
    Préprocesseur MMEA (sensor/csv + video/mp4)
    - Crée fenêtres IMU
    - Calcule start_frame estimé (si tu veux extraire un clip vidéo aligné)
    - Sauvegarde windows + metadata CSV
    """

    def __init__(self, config: Config):
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
            "bad_format_paths": 0,
            "missing_sensor_files": 0,
            "missing_video_files": 0,
            "prefix_mismatch": 0,
        }

    # ---------- splits ----------
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

        samples: List[str] = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and (not line.startswith("#")):
                    samples.append(line)

        print(f"[{split}] Chargé {len(samples)} échantillons depuis {split_file}")
        return samples

    # ---------- parsing paths ----------
    def parse_sensor_relpath(self, sensor_relpath: str) -> Dict:
        """
        Attendu (relatif) : sensor/1_upstairs/1_upstairs_2020_12_15_15_58_48.csv
        """
        p = Path(sensor_relpath)
        parts = p.parts

        # Doit contenir 'sensor'
        if "sensor" not in parts:
            self.preprocessing_stats["bad_format_paths"] += 1
            raise ValueError(f"Chemin invalide (pas de dossier 'sensor'): {sensor_relpath}")

        # class_dir = élément juste après 'sensor'
        sensor_idx = parts.index("sensor")
        if len(parts) < sensor_idx + 2:
            self.preprocessing_stats["bad_format_paths"] += 1
            raise ValueError(f"Chemin invalide (pas de class_dir): {sensor_relpath}")

        class_dir = parts[sensor_idx + 1]  # ex: "1_upstairs"
        sample_id = p.stem                 # ex: "1_upstairs_2020_12_15_15_58_48"

        # split class_dir => class_num + class_name
        if "_" in class_dir:
            class_num_str, class_name = class_dir.split("_", 1)
        else:
            class_num_str, class_name = class_dir, class_dir

        class_num = int(class_num_str) if class_num_str.isdigit() else -1

        return {
            "class_dir": class_dir,
            "class_num": class_num,        # ex: 1
            "class_name": class_name,      # ex: upstairs
            "sample_id": sample_id,        # prefix commun
            "sensor_path": sensor_relpath,
            "user_id": "unknown",
        }

    # ---------- paths video ----------
    def sensor_to_video_relpath(self, sensor_relpath: str) -> str:
        """
        sensor/1_upstairs/xxx.csv -> video/1_upstairs/xxx.mp4
        """
        p = Path(sensor_relpath)

        # remplacer uniquement le premier dossier "sensor" par "video"
        parts = list(p.parts)
        if "sensor" not in parts:
            raise ValueError(f"Chemin invalide (pas 'sensor'): {sensor_relpath}")
        idx = parts.index("sensor")
        parts[idx] = "video"

        video_p = Path(*parts).with_suffix(self.data_cfg.video_ext)
        return str(video_p)

    def check_exists(self, relpath: str) -> bool:
        return (self.paths.base_input / relpath).exists()

    # ---------- IMU loading ----------
    def load_imu_csv(self, sensor_relpath: str) -> Optional[np.ndarray]:
        """
        Lit CSV (6 colonnes) + conversion Racc/Rgyro.
        Retourne np.ndarray shape (N, 6) float32, ou None si introuvable/erreur.
        """
        full_path = self.paths.base_input / sensor_relpath
        if not full_path.exists():
            self.preprocessing_stats["missing_sensor_files"] += 1
            return None

        try:
            data = pd.read_csv(full_path, header=None).values.astype(np.float32)

            if data.ndim == 1:
                data = data.reshape(1, -1)

            # Normaliser le nb de colonnes à 6
            if data.shape[1] < 6:
                pad = np.zeros((data.shape[0], 6 - data.shape[1]), dtype=np.float32)
                data = np.hstack([data, pad])
            elif data.shape[1] > 6:
                data = data[:, :6]

            # Conversion physique (dataset note)
            acc = data[:, :3] / float(self.data_cfg.Racc)
            gyro = data[:, 3:6] / float(self.data_cfg.Rgyro)
            data = np.concatenate([acc, gyro], axis=1).astype(np.float32)

            return data
        except Exception:
            # fichier corrompu / format bizarre => skip
            return None

    # ---------- signal ops ----------
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
        # 1) median filter
        k = int(self.data_cfg.median_filter_kernel)
        if k and k > 1:
            # scipy medfilt demande k impair
            if k % 2 == 0:
                k += 1
            filtered = np.zeros_like(imu, dtype=np.float32)
            for ch in range(imu.shape[1]):
                filtered[:, ch] = signal.medfilt(imu[:, ch], kernel_size=k)
            imu = filtered

        # 2) z-score par canal (sur ce sample)
        if self.data_cfg.normalize_imu:
            mean = imu.mean(axis=0, keepdims=True)
            std = imu.std(axis=0, keepdims=True) + 1e-8
            imu = (imu - mean) / std

        return imu.astype(np.float32)

    def create_imu_windows(self, imu: np.ndarray) -> List[np.ndarray]:
        ws = int(self.data_cfg.imu_window_size)
        stride = int(self.data_cfg.imu_stride)
        n = imu.shape[0]

        if n < ws:
            if self.data_cfg.pad_short_sequences:
                pad = np.zeros((ws - n, imu.shape[1]), dtype=np.float32)
                imu = np.vstack([imu, pad])
                n = ws
            else:
                return []  # skip

        windows = []
        for start in range(0, n - ws + 1, stride):
            windows.append(imu[start:start + ws])
        return windows

    # ---------- alignment helpers ----------
    def estimate_start_frame(self, window_idx: int) -> int:
        """
        Estime le start_frame vidéo correspondant à la window IMU.
        Hypothèse : IMU et vidéo sont synchronisés et commencent au même t=0.
        """
        sr = float(self.data_cfg.imu_sampling_rate)
        stride = float(self.data_cfg.imu_stride)
        start_time_sec = window_idx * (stride / sr)
        return int(round(start_time_sec * float(self.data_cfg.video_fps)))

    # ---------- main split processing ----------
    def preprocess_split(self, split: str, save: bool = True) -> pd.DataFrame:
        samples = self.load_split(split)
        self.preprocessing_stats["total_samples"] += len(samples)

        records: List[Dict] = []

        print("\n" + "=" * 60)
        print(f"Preprocessing du split: {split.upper()}")
        print("=" * 60)

        for sensor_relpath in tqdm(samples, desc=f"Processing {split}"):
            # 1) parse + construire chemins
            try:
                info = self.parse_sensor_relpath(sensor_relpath)
            except ValueError:
                self.preprocessing_stats["skipped_samples"] += 1
                continue

            self.preprocessing_stats["classes_found"].add(info["class_dir"])

            video_relpath = self.sensor_to_video_relpath(sensor_relpath)

            sensor_ok = self.check_exists(sensor_relpath)
            video_ok = self.check_exists(video_relpath)

            if not sensor_ok:
                self.preprocessing_stats["missing_sensor_files"] += 1
                self.preprocessing_stats["skipped_samples"] += 1
                continue

            if video_ok:
                self.preprocessing_stats["samples_with_video"] += 1
            else:
                self.preprocessing_stats["samples_without_video"] += 1
                self.preprocessing_stats["missing_video_files"] += 1
                # Tu peux décider de skip les samples sans vidéo si tu fais du cross-modal strict
                # Ici, on garde quand même la metadata, mais les windows auront video_exists=False

            # 2) vérifier prefix (même sample_id)
            #    vidéo et csv doivent partager le même stem
            if Path(video_relpath).stem != Path(sensor_relpath).stem:
                self.preprocessing_stats["prefix_mismatch"] += 1
                self.preprocessing_stats["skipped_samples"] += 1
                continue

            # 3) load imu
            imu_raw = self.load_imu_csv(sensor_relpath)
            if imu_raw is None or imu_raw.size == 0:
                self.preprocessing_stats["skipped_samples"] += 1
                continue

            # 4) resample si on connait original_rate
            orig_rate = self.data_cfg.imu_original_rate
            if orig_rate is not None:
                imu_raw = self.resample_imu(
                    imu_raw,
                    original_rate=float(orig_rate),
                    target_rate=float(self.data_cfg.imu_sampling_rate),
                )

            # 5) preprocess
            imu_proc = self.preprocess_imu(imu_raw)

            # 6) windows
            windows = self.create_imu_windows(imu_proc)
            if (not windows) and (not self.data_cfg.pad_short_sequences):
                self.preprocessing_stats["skipped_samples"] += 1
                continue

            # 7) label basé sur class_num (plus fiable que mapping texte)
            #    si tes classes commencent à 1, tu peux faire label = class_num - 1
            label = info["class_num"]

            # 8) save windows + records
            for w_idx, window in enumerate(windows):
                self.preprocessing_stats["total_windows"] += 1

                record = {
                    "split": split,
                    "user_id": info["user_id"],
                    "class_dir": info["class_dir"],
                    "class_name": info["class_name"],
                    "class_num": info["class_num"],
                    "label": label,
                    "sample_id": info["sample_id"],
                    "window_idx": w_idx,
                    "sensor_path": sensor_relpath,
                    "video_path": video_relpath,
                    "video_exists": video_ok,
                    "start_frame": self.estimate_start_frame(w_idx),
                    "imu_shape_0": int(window.shape[0]),
                    "imu_shape_1": int(window.shape[1]),
                }

                if save:
                    out_dir = self.paths.preprocessed_dir / split
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # nom stable : classdir + sample_id + window
                    window_filename = f"{info['class_dir']}_{info['sample_id']}_w{w_idx}.npy"
                    window_path = out_dir / window_filename
                    np.save(window_path, window.astype(np.float32))
                    record["imu_window_path"] = str(window_path)

                records.append(record)

        df = pd.DataFrame(records)

        if save:
            self.paths.preprocessed_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self.paths.preprocessed_dir / f"{split}_metadata.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Metadata sauvegardée: {csv_path}")
            if len(df) > 0:
                print(f"  Total windows: {len(df)}")
                print(f"  Windows avec vidéo: {int(df['video_exists'].sum())}")
                print(f"  Windows sans vidéo: {int((~df['video_exists']).sum())}")
            else:
                print("  ⚠️ Aucun record généré (vérifie les chemins/splits).")

        return df

    def run_full_preprocessing(self) -> Dict[str, pd.DataFrame]:
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLET DU DATASET MMEA")
        print("=" * 60)

        results: Dict[str, pd.DataFrame] = {}

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
        print(f"Bad format paths: {self.preprocessing_stats['bad_format_paths']}")
        print(f"Missing sensor: {self.preprocessing_stats['missing_sensor_files']}")
        print(f"Missing video: {self.preprocessing_stats['missing_video_files']}")
        print(f"Prefix mismatch: {self.preprocessing_stats['prefix_mismatch']}")

        # save stats
        stats_path = self.paths.preprocessed_dir / "preprocessing_stats.json"
        stats_to_save = dict(self.preprocessing_stats)
        stats_to_save["classes_found"] = sorted(list(stats_to_save["classes_found"]))
        self.paths.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_to_save, f, indent=2)
        print(f"\n✓ Statistiques sauvegardées: {stats_path}")
        print("\n" + "=" * 60)
        print("PREPROCESSING TERMINÉ")
        print("=" * 60)

        return results


# -----------------------------
# main (exemple)
# -----------------------------
def main():
    # Exemple minimal : adapte tes chemins Kaggle
    base_input = Path("/kaggle/input/dataset-har/UESTC-MMEA-CL")  # <-- adapte si besoin
    preproc_dir = Path("./preprocessed_mmea")                     # <-- sortie

    cfg = Config(
        paths=PathsConfig(
            base_input=base_input,
            preprocessed_dir=preproc_dir,
            train_file="train.txt",
            val_file="val.txt",
            test_file="test.txt",
        ),
        data=DataConfig(
            imu_sampling_rate=50.0,
            imu_original_rate=None,   # mets 50.0 ou 100.0 si tu connais la fréquence réelle
            imu_window_size=250,
            imu_stride=125,
            median_filter_kernel=5,
            normalize_imu=True,
            video_fps=25.0,
            video_ext=".mp4",
            Racc=16384.0,
            Rgyro=16.4,
            pad_short_sequences=True,
        )
    )

    print("Test du preprocessor MMEA")
    print(f"Dataset path: {cfg.paths.base_input}")

    if not cfg.paths.base_input.exists():
        print(f"ERREUR: Dataset introuvable à {cfg.paths.base_input}")
        return

    preprocessor = MMEAPreprocessor(cfg)
    preprocessor.run_full_preprocessing()
    print("\n✓ Terminé.")


if __name__ == "__main__":
    main()
