"""
Preprocessing complet du dataset MMEA pour IMU-Video Cross-modal Learning
Conforme à l'article: fenêtres de 5s à 50Hz, median filter, z-score normalization
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import scipy.signal as signal
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class MMEAPreprocessor:
    """
    Préprocesseur pour dataset MMEA
    - Charge les données IMU et vidéo
    - Crée des fenêtres de 5 secondes (250 timestamps à 50Hz)
    - Applique median filtering et z-score normalization
    - Associe les clips vidéo correspondants
    """
    
    def __init__(self, config):
        self.config = config
        self.paths = config.paths
        self.data_cfg = config.data
        
        # Mapping classe -> indice
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Stats pour rapport
        self.preprocessing_stats = {
            'total_samples': 0,
            'total_windows': 0,
            'samples_with_video': 0,
            'samples_without_video': 0,
            'classes_found': set()
        }
    
    def load_split(self, split: str) -> List[str]:
        """
        Charge un split (train/val/test)
        Format attendu: chaque ligne = chemin relatif vers fichier sensor
        Exemple: sensor/1/drink_water/1.txt
        """
        if split == 'train':
            split_file = self.paths.base_input / self.paths.train_file
        elif split == 'val':
            split_file = self.paths.base_input / self.paths.val_file
        elif split == 'test':
            split_file = self.paths.base_input / self.paths.test_file
        else:
            raise ValueError(f"Split inconnu: {split}")
        
        if not split_file.exists():
            raise FileNotFoundError(f"Fichier split introuvable: {split_file}")
        
        samples = []
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Ignorer lignes vides et commentaires
                    samples.append(line)
        
        print(f"[{split}] Chargé {len(samples)} échantillons depuis {split_file}")
        return samples
    
    def parse_sample_path(self, sample_path: str) -> Dict:
        """
        Parse le chemin pour extraire: user_id, class_name, sample_id
        Format MMEA: sensor/[user_id]/[class_name]/[sample_id].txt
        Exemple: sensor/1/drink_water/1.txt
        """
        parts = Path(sample_path).parts
        
        try:
            # Trouver l'index de 'sensor'
            sensor_idx = parts.index('sensor')
            
            if len(parts) >= sensor_idx + 4:
                user_id = parts[sensor_idx + 1]
                class_name = parts[sensor_idx + 2]
                sample_id = Path(parts[sensor_idx + 3]).stem
            else:
                raise ValueError(f"Format de chemin invalide: {sample_path}")
            
        except (ValueError, IndexError) as e:
            # Fallback: essayer de parser autrement
            print(f"Avertissement: parsing fallback pour {sample_path}")
            user_id = "unknown"
            class_name = Path(sample_path).parent.name
            sample_id = Path(sample_path).stem
        
        return {
            'user_id': user_id,
            'class_name': class_name,
            'sample_id': sample_id,
            'sensor_path': sample_path
        }
    
    def load_imu_data(self, sensor_path: str) -> np.ndarray:
        """
        Charge les données IMU depuis un fichier texte
        Format attendu: N lignes x 6 colonnes (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
        """
        full_path = self.paths.base_input / sensor_path
        
        if not full_path.exists():
            print(f"Avertissement: fichier IMU introuvable: {full_path}")
            return np.zeros((100, 6))  # Retourner données vides
        
        try:
            # Essayer de charger avec numpy
            data = np.loadtxt(full_path)
            
            # Vérifier dimensions
            if data.ndim == 1:
                # Si 1D, supposer que c'est une seule ligne
                data = data.reshape(1, -1)
            
            # Vérifier nombre de colonnes
            if data.shape[1] < 6:
                print(f"Avertissement: {sensor_path} a {data.shape[1]} colonnes < 6")
                # Padding avec zeros
                padding = np.zeros((data.shape[0], 6 - data.shape[1]))
                data = np.hstack([data, padding])
            elif data.shape[1] > 6:
                print(f"Avertissement: {sensor_path} a {data.shape[1]} colonnes > 6, troncature")
                data = data[:, :6]
            
            return data.astype(np.float32)
        
        except Exception as e:
            print(f"Erreur lors du chargement de {sensor_path}: {e}")
            return np.zeros((100, 6), dtype=np.float32)
    
    def resample_imu(self, imu_data: np.ndarray, original_rate: float, target_rate: float = 50.0) -> np.ndarray:
        """
        Resample IMU data à la fréquence cible (50Hz par défaut)
        """
        if original_rate == target_rate:
            return imu_data
        
        num_samples = imu_data.shape[0]
        num_target_samples = int(num_samples * target_rate / original_rate)
        
        # Resample chaque canal indépendamment
        resampled = []
        for channel in range(imu_data.shape[1]):
            resampled_channel = signal.resample(imu_data[:, channel], num_target_samples)
            resampled.append(resampled_channel)
        
        return np.stack(resampled, axis=1).astype(np.float32)
    
    def preprocess_imu(self, imu_data: np.ndarray) -> np.ndarray:
        """
        Applique le preprocessing IMU conforme à l'article:
        1. Median filtering (kernel size = 5)
        2. Z-score normalization (par canal)
        """
        # 1) Median filtering pour réduction du bruit
        if self.data_cfg.median_filter_kernel > 1:
            # Appliquer median filter sur chaque canal
            filtered = np.zeros_like(imu_data)
            for channel in range(imu_data.shape[1]):
                filtered[:, channel] = signal.medfilt(
                    imu_data[:, channel],
                    kernel_size=self.data_cfg.median_filter_kernel
                )
            imu_data = filtered
        
        # 2) Z-score normalization (par canal)
        if self.data_cfg.normalize_imu:
            mean = np.mean(imu_data, axis=0, keepdims=True)
            std = np.std(imu_data, axis=0, keepdims=True) + 1e-8  # Éviter division par zéro
            imu_data = (imu_data - mean) / std
        
        return imu_data.astype(np.float32)
    
    def create_imu_windows(self, imu_data: np.ndarray) -> List[np.ndarray]:
        """
        Crée des fenêtres glissantes sur les données IMU
        Fenêtres de 250 timestamps (5 secondes à 50Hz) avec stride configurable
        Returns: list of windows, shape (window_size, channels)
        """
        windows = []
        window_size = self.data_cfg.imu_window_size  # 250
        stride = self.data_cfg.imu_stride  # 125 par défaut (50% overlap)
        
        num_samples = imu_data.shape[0]
        
        # Vérifier si assez de données
        if num_samples < window_size:
            # Padding si pas assez de données
            padding = np.zeros((window_size - num_samples, imu_data.shape[1]), dtype=np.float32)
            imu_data = np.vstack([imu_data, padding])
            num_samples = window_size
        
        # Créer les fenêtres
        for start in range(0, num_samples - window_size + 1, stride):
            window = imu_data[start:start + window_size]
            windows.append(window)
        
        return windows
    
    def get_video_path(self, sensor_path: str) -> str:
        """
        Construit le chemin vidéo correspondant au sensor path
        Exemple: sensor/1/drink_water/1.txt -> video/1/drink_water/1.avi
        """
        # Remplacer 'sensor' par 'video' et extension .txt par .avi
        video_path = sensor_path.replace('sensor', 'video')
        
        # Changer extension
        video_path = Path(video_path).with_suffix('.avi')
        
        return str(video_path)
    
    def check_video_exists(self, video_path: str) -> bool:
        """Vérifie si le fichier vidéo existe"""
        full_path = self.paths.base_input / video_path
        return full_path.exists()
    
    def build_class_mapping(self, all_samples: List[str]):
        """
        Construit le mapping classe -> index à partir de tous les samples
        """
        classes = set()
        for sample in all_samples:
            info = self.parse_sample_path(sample)
            classes.add(info['class_name'])
        
        # Trier pour cohérence
        classes = sorted(list(classes))
        
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        
        print(f"\n✓ Trouvé {len(classes)} classes:")
        for i, c in enumerate(classes):
            if i < 10:  # Afficher seulement les 10 premières
                print(f"  {i}: {c}")
        if len(classes) > 10:
            print(f"  ... et {len(classes) - 10} autres")
    
    def preprocess_split(self, split: str, save: bool = True) -> pd.DataFrame:
        """
        Prétraite un split complet et génère un DataFrame avec les métadonnées
        
        Returns:
            DataFrame avec colonnes:
            - split, user_id, class_name, label, sample_id, window_idx
            - sensor_path, video_path, video_exists
            - imu_window_path (chemin vers .npy sauvegardé)
            - start_frame (pour extraire le clip vidéo correspondant)
        """
        samples = self.load_split(split)
        self.preprocessing_stats['total_samples'] += len(samples)
        
        records = []
        
        print(f"\n{'='*60}")
        print(f"Preprocessing du split: {split.upper()}")
        print(f"{'='*60}")
        
        for sample_path in tqdm(samples, desc=f"Processing {split}"):
            # Parse info
            info = self.parse_sample_path(sample_path)
            self.preprocessing_stats['classes_found'].add(info['class_name'])
            
            # Charger IMU
            imu_raw = self.load_imu_data(sample_path)
            
            # Resample à 50Hz (supposer que MMEA est à 25Hz initialement)
            # Note: ajuster original_rate selon vos données réelles
            imu_resampled = self.resample_imu(imu_raw, original_rate=25.0, target_rate=50.0)
            
            # Preprocessing
            imu_preprocessed = self.preprocess_imu(imu_resampled)
            
            # Créer windows
            windows = self.create_imu_windows(imu_preprocessed)
            
            # Video path
            video_path = self.get_video_path(sample_path)
            video_exists = self.check_video_exists(video_path)
            
            if video_exists:
                self.preprocessing_stats['samples_with_video'] += 1
            else:
                self.preprocessing_stats['samples_without_video'] += 1
            
            # Label
            class_name = info['class_name']
            label = self.class_to_idx.get(class_name, -1)
            
            # Sauvegarder chaque window
            for i, window in enumerate(windows):
                self.preprocessing_stats['total_windows'] += 1
                
                # Calculer start_frame pour la vidéo
                # À 25 FPS vidéo et stride de 125 samples (2.5s à 50Hz)
                video_fps = self.data_cfg.video_fps
                start_time = i * (self.data_cfg.imu_stride / self.data_cfg.imu_sampling_rate)
                start_frame = int(start_time * video_fps)
                
                record = {
                    'split': split,
                    'user_id': info['user_id'],
                    'class_name': class_name,
                    'label': label,
                    'sample_id': info['sample_id'],
                    'window_idx': i,
                    'sensor_path': sample_path,
                    'video_path': video_path,
                    'video_exists': video_exists,
                    'start_frame': start_frame,
                    'imu_shape_0': window.shape[0],
                    'imu_shape_1': window.shape[1],
                }
                
                # Sauvegarder le window IMU
                if save:
                    # Créer répertoire
                    window_save_dir = self.paths.preprocessed_dir / split
                    window_save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Nom de fichier unique
                    window_filename = f"user{info['user_id']}_{class_name}_{info['sample_id']}_w{i}.npy"
                    window_save_path = window_save_dir / window_filename
                    
                    np.save(window_save_path, window)
                    record['imu_window_path'] = str(window_save_path)
                
                records.append(record)
        
        # Créer DataFrame
        df = pd.DataFrame(records)
        
        # Sauvegarder metadata
        if save:
            csv_path = self.paths.preprocessed_dir / f"{split}_metadata.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Metadata sauvegardée: {csv_path}")
            print(f"  Total windows: {len(df)}")
            print(f"  Windows avec vidéo: {df['video_exists'].sum()}")
            print(f"  Windows sans vidéo: {(~df['video_exists']).sum()}")
        
        return df
    
    def run_full_preprocessing(self):
        """
        Lance le preprocessing complet sur tous les splits
        """
        print("\n" + "="*60)
        print("PREPROCESSING COMPLET DU DATASET MMEA")
        print("="*60)
        
        # 1. Charger tous les samples pour construire class mapping
        print("\nÉtape 1: Construction du mapping des classes...")
        all_samples = []
        for split in ['train', 'val', 'test']:
            try:
                samples = self.load_split(split)
                all_samples.extend(samples)
            except FileNotFoundError:
                print(f"Avertissement: split '{split}' introuvable, ignoré")
        
        self.build_class_mapping(all_samples)
        
        # Sauvegarder le mapping
        mapping_path = self.paths.preprocessed_dir / 'class_mapping.json'
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(mapping_path, 'w') as f:
            json.dump({
                'class_to_idx': self.class_to_idx,
                'idx_to_class': self.idx_to_class,
                'num_classes': len(self.class_to_idx)
            }, f, indent=2)
        print(f"✓ Mapping classes sauvegardé: {mapping_path}")
        
        # 2. Preprocess chaque split
        print("\nÉtape 2: Preprocessing des splits...")
        results = {}
        
        for split in ['train', 'val', 'test']:
            try:
                df = self.preprocess_split(split, save=True)
                results[split] = df
            except FileNotFoundError:
                print(f"Avertissement: split '{split}' introuvable, ignoré")
                continue
        
        # 3. Afficher statistiques finales
        print("\n" + "="*60)
        print("STATISTIQUES FINALES")
        print("="*60)
        print(f"Total échantillons: {self.preprocessing_stats['total_samples']}")
        print(f"Total windows créées: {self.preprocessing_stats['total_windows']}")
        print(f"Échantillons avec vidéo: {self.preprocessing_stats['samples_with_video']}")
        print(f"Échantillons sans vidéo: {self.preprocessing_stats['samples_without_video']}")
        print(f"Classes trouvées: {len(self.preprocessing_stats['classes_found'])}")
        
        print("\nDistribution par split:")
        for split, df in results.items():
            print(f"  {split.upper()}:")
            print(f"    - Windows: {len(df)}")
            print(f"    - Classes: {df['class_name'].nunique()}")
            print(f"    - Users: {df['user_id'].nunique()}")
            
            # Top 5 classes
            print(f"    - Top 5 classes:")
            for class_name, count in df['class_name'].value_counts().head(5).items():
                print(f"      • {class_name}: {count}")
        
        # Sauvegarder stats
        stats_path = self.paths.preprocessed_dir / 'preprocessing_stats.json'
        with open(stats_path, 'w') as f:
            stats_to_save = self.preprocessing_stats.copy()
            stats_to_save['classes_found'] = list(stats_to_save['classes_found'])
            json.dump(stats_to_save, f, indent=2)
        print(f"\n✓ Statistiques sauvegardées: {stats_path}")
        
        print("\n" + "="*60)
        print("PREPROCESSING TERMINÉ AVEC SUCCÈS!")
        print("="*60)
        
        return results


def main():
    """Fonction principale pour test standalone"""
    import sys
    sys.path.append('.')
    
    from config import CONFIG
    
    print("Test du preprocessor MMEA")
    print(f"Dataset path: {CONFIG.paths.base_input}")
    
    # Vérifier que le dataset existe
    if not CONFIG.paths.base_input.exists():
        print(f"ERREUR: Dataset introuvable à {CONFIG.paths.base_input}")
        print("Veuillez ajuster les chemins dans config.py")
        return
    
    # Créer preprocessor
    preprocessor = MMEAPreprocessor(CONFIG)
    
    # Lancer preprocessing
    results = preprocessor.run_full_preprocessing()
    
    print("\n✓ Test terminé avec succès!")


if __name__ == "__main__":
    main()