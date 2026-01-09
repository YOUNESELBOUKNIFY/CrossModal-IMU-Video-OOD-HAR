"""
Notebook Kaggle principal pour IMU-Video Cross-modal HAR
À exécuter cellule par cellule dans Kaggle
"""

# ============================================
# CELL 1: Installation et imports
# ============================================

# Installer dépendances si nécessaire
# !pip install scikit-learn matplotlib seaborn opencv-python tqdm -q

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Vérifier l'environnement
IS_KAGGLE = os.path.exists('/kaggle')
print(f"Environnement: {'Kaggle' if IS_KAGGLE else 'Local'}")

# Imports standards
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Set seeds pour reproductibilité
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================
# CELL 2: Configuration
# ============================================

# Charger la configuration
from config import CONFIG

# Afficher les chemins
print(f"Base input: {CONFIG.paths.base_input}")
print(f"Base output: {CONFIG.paths.base_output}")
print(f"Preprocessed dir: {CONFIG.paths.preprocessed_dir}")

# Vérifier que les données existent
assert CONFIG.paths.base_input.exists(), "Dataset non trouvé!"
print("✓ Dataset trouvé")

# ============================================
# CELL 3: Preprocessing (à exécuter une seule fois)
# ============================================

from preprocessing import MMEAPreprocessor

# Créer le preprocessor
preprocessor = MMEAPreprocessor(CONFIG)

# Option 1: Preprocessing complet (long, ~10-30 min selon dataset)
DO_PREPROCESSING = True  # Mettre False si déjà fait

if DO_PREPROCESSING:
    print("Lancement du preprocessing...")
    results = preprocessor.run_full_preprocessing()
    
    # Afficher stats
    print("\n=== Statistiques après preprocessing ===")
    for split, df in results.items():
        print(f"\n{split.upper()}:")
        print(f"  Nombre de windows: {len(df)}")
        print(f"  Nombre de classes: {df['class_name'].nunique()}")
        print(f"  Distribution (top 10):")
        print(df['class_name'].value_counts().head(10))
else:
    print("Preprocessing skipped - chargement des métadonnées existantes...")
    train_df = pd.read_csv(CONFIG.paths.preprocessed_dir / 'train_metadata.csv')
    val_df = pd.read_csv(CONFIG.paths.preprocessed_dir / 'val_metadata.csv')
    test_df = pd.read_csv(CONFIG.paths.preprocessed_dir / 'test_metadata.csv')
    print(f"✓ Chargé: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

# ============================================
# CELL 4: Chargement des métadonnées et exploration
# ============================================

# Charger metadata
train_df = pd.read_csv(CONFIG.paths.preprocessed_dir / 'train_metadata.csv')
val_df = pd.read_csv(CONFIG.paths.preprocessed_dir / 'val_metadata.csv')
test_df = pd.read_csv(CONFIG.paths.preprocessed_dir / 'test_metadata.csv')

print(f"Train: {len(train_df)} windows")
print(f"Val: {len(val_df)} windows")
print(f"Test: {len(test_df)} windows")

# Charger class mapping
with open(CONFIG.paths.preprocessed_dir / 'class_mapping.json', 'r') as f:
    class_mapping = json.load(f)

print(f"\nNombre de classes: {len(class_mapping['class_to_idx'])}")
print(f"Classes: {list(class_mapping['class_to_idx'].keys())[:10]}...")

# Visualisation distribution des classes
plt.figure(figsize=(15, 5))
train_df['class_name'].value_counts().plot(kind='bar')
plt.title('Distribution des classes (Train)')
plt.xlabel('Classe')
plt.ylabel('Nombre de windows')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(CONFIG.paths.results_dir / 'class_distribution.png', dpi=150)
plt.show()

# ============================================
# CELL 5: Test des dataloaders
# ============================================

from datasets import CrossModalDataset, IMUClassificationDataset

# Test CrossModalDataset
print("Test CrossModalDataset...")
cm_dataset = CrossModalDataset(train_df.head(100), CONFIG)
print(f"Dataset size: {len(cm_dataset)}")

# Charger un sample
sample = cm_dataset[0]
print(f"IMU shape: {sample['imu'].shape}")  # (6, 250)
print(f"Video shape: {sample['video'].shape}")  # (10, 3, 224, 224)

# Test IMUClassificationDataset
print("\nTest IMUClassificationDataset...")
clf_dataset = IMUClassificationDataset(train_df.head(100), CONFIG)
sample_clf = clf_dataset[0]
print(f"IMU shape: {sample_clf['imu'].shape}")
print(f"Label: {sample_clf['label']}")

# Créer un petit dataloader pour test
test_loader = DataLoader(clf_dataset, batch_size=8, shuffle=False)
batch = next(iter(test_loader))
print(f"\nBatch IMU: {batch['imu'].shape}")
print(f"Batch labels: {batch['label'].shape}")

# ============================================
# CELL 6: Cross-Modal Pretraining
# ============================================

from models import CrossModalModel
from losses import SigmoidContrastiveLoss
from trainer import CrossModalTrainer
from datasets import create_dataloaders

# Créer les dataloaders
print("Création des dataloaders...")
dataloaders = create_dataloaders(
    CONFIG,
    train_df,
    val_df,
    test_df,
    mode='cross_modal'
)

# Créer le modèle
print("Création du modèle cross-modal...")
model = CrossModalModel(CONFIG)
print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")

# Loss function
loss_fn = SigmoidContrastiveLoss(learnable=True)

# Trainer
trainer = CrossModalTrainer(model, loss_fn, CONFIG, device=device)

# Entraînement
print("\n=== Début du pretraining cross-modal ===")
trainer.fit(dataloaders['train'], dataloaders['val'])

# Plot training curves
history = trainer.history
plt.figure(figsize=(10, 5))
plt.plot(history['train'], label='Train Loss')
plt.plot(history['val'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Cross-Modal Pretraining Loss')
plt.savefig(CONFIG.paths.results_dir / 'pretraining_loss.png', dpi=150)
plt.show()

print(f"\n✓ Pretraining terminé. Best val loss: {trainer.best_metric:.4f}")

# ============================================
# CELL 7: Extraction de l'encodeur pré-entraîné
# ============================================

# Charger le meilleur checkpoint
checkpoint_path = CONFIG.paths.checkpoints_dir / 'cross_modal' / 'best_model.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Extraire l'encodeur IMU
pretrained_encoder = model.imu_encoder
pretrained_encoder.eval()

print("✓ Encodeur IMU pré-entraîné extrait")

# Sauvegarder l'encodeur séparément
encoder_save_path = CONFIG.paths.checkpoints_dir / 'pretrained_imu_encoder.pt'
torch.save(pretrained_encoder.state_dict(), encoder_save_path)
print(f"✓ Encodeur sauvegardé: {encoder_save_path}")

# ============================================
# CELL 8: Classification - Linear Probing
# ============================================

from models import IMUClassifier
from trainer import ClassificationTrainer

# Créer dataloaders pour classification
clf_loaders = create_dataloaders(
    CONFIG,
    train_df,
    val_df,
    test_df,
    mode='classification'
)

# Créer classifier avec linear probing
print("\n=== Linear Probing ===")
classifier_probe = IMUClassifier(
    pretrained_encoder,
    CONFIG,
    freeze_encoder=True
)

trainer_probe = ClassificationTrainer(
    classifier_probe,
    CONFIG,
    device=device,
    mode='linear_probe'
)

# Entraînement
trainer_probe.fit(clf_loaders['train'], clf_loaders['val'])

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
train_losses = [m['loss'] for m in trainer_probe.history['train']]
val_losses = [m['loss'] for m in trainer_probe.history['val']]
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Linear Probing - Loss')

plt.subplot(1, 2, 2)
train_accs = [m['accuracy'] for m in trainer_probe.history['train']]
val_accs = [m['accuracy'] for m in trainer_probe.history['val']]
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Linear Probing - Accuracy')

plt.tight_layout()
plt.savefig(CONFIG.paths.results_dir / 'linear_probing.png', dpi=150)
plt.show()

# ============================================
# CELL 9: Classification - Full Finetuning
# ============================================

print("\n=== Full Finetuning ===")
classifier_ft = IMUClassifier(
    pretrained_encoder,
    CONFIG,
    freeze_encoder=False
)

trainer_ft = ClassificationTrainer(
    classifier_ft,
    CONFIG,
    device=device,
    mode='finetune'
)

# Entraînement
trainer_ft.fit(clf_loaders['train'], clf_loaders['val'])

# Plot similaire
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
train_losses = [m['loss'] for m in trainer_ft.history['train']]
val_losses = [m['loss'] for m in trainer_ft.history['val']]
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Full Finetuning - Loss')

plt.subplot(1, 2, 2)
train_accs = [m['balanced_accuracy'] for m in trainer_ft.history['val']]
plt.plot(train_accs, label='Val Balanced Acc')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy (%)')
plt.legend()
plt.title('Full Finetuning - Balanced Accuracy')

plt.tight_layout()
plt.savefig(CONFIG.paths.results_dir / 'finetuning.png', dpi=150)
plt.show()

# ============================================
# CELL 10: Évaluation complète sur test set
# ============================================

from evaluator import Evaluator

print("\n=== Évaluation sur Test Set ===")

# Linear probing
checkpoint_path_probe = CONFIG.paths.checkpoints_dir / 'classifier_linear_probe' / 'best_model.pt'
classifier_probe.load_state_dict(torch.load(checkpoint_path_probe)['model_state_dict'])

evaluator_probe = Evaluator(classifier_probe, CONFIG, device)
results_probe = evaluator_probe.evaluate(clf_loaders['test'])

print("\nLinear Probing Results:")
for metric, value in results_probe['metrics'].items():
    print(f"  {metric}: {value:.2f}")

# Full finetuning
checkpoint_path_ft = CONFIG.paths.checkpoints_dir / 'classifier_finetune' / 'best_model.pt'
classifier_ft.load_state_dict(torch.load(checkpoint_path_ft)['model_state_dict'])

evaluator_ft = Evaluator(classifier_ft, CONFIG, device)
results_ft = evaluator_ft.evaluate(clf_loaders['test'])

print("\nFull Finetuning Results:")
for metric, value in results_ft['metrics'].items():
    print(f"  {metric}: {value:.2f}")

# Créer tableau comparatif
comparison = pd.DataFrame({
    'Linear Probing': results_probe['metrics'],
    'Full Finetuning': results_ft['metrics']
}).T

print("\n=== Tableau Comparatif ===")
print(comparison.to_string())

# Sauvegarder
comparison.to_csv(CONFIG.paths.results_dir / 'comparison_probe_vs_finetune.csv')

# ============================================
# CELL 11: Confusion Matrix
# ============================================

# Plot confusion matrix pour meilleur modèle
evaluator_ft.plot_confusion_matrix(
    results_ft['labels'],
    results_ft['predictions'],
    class_names=list(class_mapping['class_to_idx'].keys()),
    save_path=CONFIG.paths.results_dir / 'confusion_matrix.png'
)

# ============================================
# CELL 12: Few-Shot Evaluation
# ============================================

from evaluator import FewShotEvaluator

print("\n=== Few-Shot Evaluation ===")

# Réduire le nombre de runs pour Kaggle (sinon trop long)
CONFIG.eval.few_shot_runs = 3
CONFIG.eval.few_shot_samples = [10, 20, 50, 100]

fs_evaluator = FewShotEvaluator(CONFIG, device)

# Run experiments
fs_results = fs_evaluator.run_few_shot_experiments(
    pretrained_encoder,
    train_df,
    test_df,
    experiment_name="cross_modal_pretrained"
)

# Agréger résultats
fs_agg = fs_evaluator.aggregate_results(fs_results)

print("\n=== Few-Shot Results (Aggregated) ===")
print(fs_agg.to_string())

# Sauvegarder
fs_results.to_csv(CONFIG.paths.results_dir / 'fewshot_results_raw.csv', index=False)
fs_agg.to_csv(CONFIG.paths.results_dir / 'fewshot_results_aggregated.csv', index=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for mode, ax in zip(['linear_probe', 'finetune'], axes):
    subset = fs_agg[fs_agg['mode'] == mode]
    ax.errorbar(
        subset['n_samples'],
        subset['balanced_accuracy_mean'],
        yerr=subset['balanced_accuracy_std'],
        marker='o', capsize=5
    )
    ax.set_xlabel('Samples per class')
    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_title(f'Few-Shot: {mode}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(CONFIG.paths.results_dir / 'fewshot_performance.png', dpi=150)
plt.show()

# ============================================
# CELL 13: Génération du rapport final
# ============================================

print("\n" + "="*60)
print("RAPPORT FINAL")
print("="*60)

# Résumé
report = {
    'Pretraining': {
        'Best Val Loss': trainer.best_metric,
        'Total Epochs': trainer.current_epoch
    },
    'Linear Probing': {
        'Test Accuracy': results_probe['metrics']['accuracy'],
        'Test Balanced Accuracy': results_probe['metrics']['balanced_accuracy'],
        'Test F1 Macro': results_probe['metrics']['f1_macro']
    },
    'Full Finetuning': {
        'Test Accuracy': results_ft['metrics']['accuracy'],
        'Test Balanced Accuracy': results_ft['metrics']['balanced_accuracy'],
        'Test F1 Macro': results_ft['metrics']['f1_macro']
    },
    'Few-Shot (100 samples)': {
        'Linear Probe Bal. Acc': fs_agg[
            (fs_agg['mode']=='linear_probe') & (fs_agg['n_samples']==100)
        ]['balanced_accuracy_mean'].values[0],
        'Finetune Bal. Acc': fs_agg[
            (fs_agg['mode']=='finetune') & (fs_agg['n_samples']==100)
        ]['balanced_accuracy_mean'].values[0]
    }
}

# Afficher
for section, metrics in report.items():
    print(f"\n{section}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.2f}")

# Sauvegarder en JSON
report_path = CONFIG.paths.results_dir / 'final_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Rapport final sauvegardé: {report_path}")

# Liste tous les fichiers générés
print("\n=== Fichiers générés ===")
for file in sorted(CONFIG.paths.results_dir.glob('*')):
    print(f"  {file.name}")

print("\n" + "="*60)
print("PIPELINE COMPLET TERMINÉ!")
print("="*60)