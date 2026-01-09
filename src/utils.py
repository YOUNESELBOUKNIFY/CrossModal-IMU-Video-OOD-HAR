"""
Fonctions utilitaires pour le projet
"""
import torch
import numpy as np
import random
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed=42):
    """
    Set random seeds pour reproductibilité
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Compte le nombre de paramètres dans un modèle
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def print_model_info(model, name="Model"):
    """
    Affiche des informations sur le modèle
    """
    params = count_parameters(model)
    print(f"\n{name} Information:")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    print(f"  Frozen parameters: {params['frozen']:,}")


def save_config(config, path):
    """
    Sauvegarde une configuration
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    
    print(f"Config sauvegardée: {path}")


def load_checkpoint(model, checkpoint_path, device='cuda', strict=True):
    """
    Charge un checkpoint dans un modèle
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    
    print(f"Checkpoint chargé: {checkpoint_path}")
    
    return checkpoint


def plot_training_curves(history, save_path=None):
    """
    Plot les courbes d'entraînement
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    if isinstance(history['train'][0], dict):
        train_losses = [m['loss'] for m in history['train']]
        val_losses = [m['loss'] for m in history['val']]
    else:
        train_losses = history['train']
        val_losses = history['val']
    
    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Training Loss')
    
    # Accuracy (si disponible)
    if isinstance(history['train'][0], dict) and 'accuracy' in history['train'][0]:
        train_accs = [m['accuracy'] for m in history['train']]
        val_accs = [m['accuracy'] for m in history['val']]
        
        axes[1].plot(train_accs, label='Train')
        axes[1].plot(val_accs, label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Training Accuracy')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Courbes sauvegardées: {save_path}")
    
    plt.show()


def format_metric_table(metrics_dict):
    """
    Formate un dictionnaire de métriques en tableau lisible
    """
    import pandas as pd
    
    df = pd.DataFrame(metrics_dict).T
    df = df.round(2)
    
    return df


def create_latex_table(df, caption="Results", label="tab:results"):
    """
    Crée un tableau LaTeX à partir d'un DataFrame
    """
    latex_str = df.to_latex(
        float_format="%.2f",
        caption=caption,
        label=label,
        escape=False
    )
    
    return latex_str


def visualize_imu_window(imu_data, title="IMU Window", save_path=None):
    """
    Visualise une fenêtre IMU
    Args:
        imu_data: numpy array de shape (window_size, 6) ou (6, window_size)
    """
    if imu_data.shape[0] == 6:
        imu_data = imu_data.T
    
    channel_names = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Accelerometer
    for i in range(3):
        axes[0].plot(imu_data[:, i], label=channel_names[i])
    axes[0].set_ylabel('Acceleration')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'{title} - Accelerometer')
    
    # Gyroscope
    for i in range(3, 6):
        axes[1].plot(imu_data[:, i], label=channel_names[i])
    axes[1].set_ylabel('Gyroscope')
    axes[1].set_xlabel('Time steps')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f'{title} - Gyroscope')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def get_device():
    """
    Retourne le device disponible (cuda ou cpu)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("GPU non disponible, utilisation du CPU")
    
    return device


def check_dataset_paths(config):
    """
    Vérifie que tous les chemins du dataset existent
    """
    paths_ok = True
    
    # Base input
    if not config.paths.base_input.exists():
        print(f"❌ Dataset introuvable: {config.paths.base_input}")
        paths_ok = False
    else:
        print(f"✓ Dataset trouvé: {config.paths.base_input}")
    
    # Splits
    for split in ['train.txt', 'val.txt', 'test.txt']:
        split_path = config.paths.base_input / split
        if not split_path.exists():
            print(f"❌ Split introuvable: {split_path}")
            paths_ok = False
        else:
            print(f"✓ Split trouvé: {split}")
    
    # Sensor et video dirs
    sensor_dir = config.paths.base_input / config.paths.sensor_dir
    video_dir = config.paths.base_input / config.paths.video_dir
    
    if not sensor_dir.exists():
        print(f"❌ Sensor dir introuvable: {sensor_dir}")
        paths_ok = False
    else:
        print(f"✓ Sensor dir trouvé")
    
    if not video_dir.exists():
        print(f"⚠️  Video dir introuvable: {video_dir} (optionnel)")
    else:
        print(f"✓ Video dir trouvé")
    
    return paths_ok


def estimate_training_time(num_samples, batch_size, num_epochs, time_per_batch=0.5):
    """
    Estime le temps d'entraînement
    Args:
        time_per_batch: temps en secondes par batch (estimation)
    """
    num_batches_per_epoch = num_samples / batch_size
    total_batches = num_batches_per_epoch * num_epochs
    total_time_seconds = total_batches * time_per_batch
    
    hours = int(total_time_seconds // 3600)
    minutes = int((total_time_seconds % 3600) // 60)
    
    print(f"\nEstimation du temps d'entraînement:")
    print(f"  Nombre de batches par epoch: {num_batches_per_epoch:.0f}")
    print(f"  Total batches: {total_batches:.0f}")
    print(f"  Temps estimé: {hours}h {minutes}min")


def create_results_summary(results_dir):
    """
    Crée un résumé de tous les résultats
    """
    results_dir = Path(results_dir)
    
    summary = {
        'csv_files': list(results_dir.glob('*.csv')),
        'json_files': list(results_dir.glob('*.json')),
        'png_files': list(results_dir.glob('*.png')),
    }
    
    print("\n=== Résumé des résultats ===")
    for file_type, files in summary.items():
        print(f"\n{file_type}:")
        for f in files:
            print(f"  - {f.name}")
    
    return summary


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(seconds):
    """
    Formate un temps en secondes en format lisible
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


if __name__ == "__main__":
    print("Module utils.py - Prêt")