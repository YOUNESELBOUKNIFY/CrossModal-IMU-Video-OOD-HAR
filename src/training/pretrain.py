import sys
import os

# <--- MODIFICATION 1 : AJOUTER CE BLOC TOUT EN HAUT ---
# Cela force Python à voir le dossier racine du projet pour les imports 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# ------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

# Maintenant les imports fonctionneront sans erreur
from src.models.contrastive import CrossModalContrastiveModel
from src.data.dataset import MMEADataset 

def load_config(config_path):
    # On sécurise le chemin du config
    full_path = os.path.join(project_root, config_path)
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, desc="Training")
    
    for batch_idx, (imu, video) in enumerate(loop):
        imu = imu.to(device)
        video = video.to(device)
        
        optimizer.zero_grad()
        
        imu_feat, video_feat = model(imu, video)
        
        if isinstance(model, nn.DataParallel):
            loss = model.module.compute_loss(imu_feat, video_feat)
        else:
            loss = model.compute_loss(imu_feat, video_feat)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def main():
    # 1. Chargement de la config
    cfg = load_config('configs/config.yaml')
    
    # 2. Configuration des GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device('cuda')
        print(f"✅ {num_gpus} GPU(s) détecté(s) !")
        if num_gpus > 1:
            print(f"   Les GPUs seront utilisés en parallèle.")
    else:
        device = torch.device('cpu')
        print("⚠️ Aucun GPU détecté.")

    # <--- MODIFICATION 2 : CHARGEMENT DES DONNÉES DEPUIS TRAIN.TXT ---
    print("📂 Lecture du fichier train.txt...")
    
    # Chemin vers votre dataset Kaggle
    dataset_root = "/kaggle/input/dataset-har/UESTC-MMEA-CL"
    train_txt_path = os.path.join(dataset_root, "train.txt")
    
    file_list = []
    if os.path.exists(train_txt_path):
        with open(train_txt_path, 'r') as f:
            # On lit chaque ligne et on enlève les espaces vides
            file_list = [line.strip() for line in f.readlines() if line.strip()]
        print(f"✅ {len(file_list)} fichiers trouvés dans train.txt")
    else:
        print(f"❌ ERREUR : Impossible de trouver {train_txt_path}")
        return # Arrêt d'urgence

    # On passe la vraie liste 'file_list' au lieu de []
    train_dataset = MMEADataset(file_list, mode='pretrain', 
                                imu_params=cfg['preprocessing']['imu'],
                                video_params=cfg['preprocessing']['video'])
    
    # -----------------------------------------------------------------
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg['training']['batch_size'], 
                              shuffle=True, 
                              num_workers=4,  
                              pin_memory=True)

    # 4. Initialisation du modèle
    print("Construction du modèle...")
    model = CrossModalContrastiveModel(cfg).to(device)
    
    if torch.cuda.device_count() > 1:
        print("🚀 Activation de DataParallel...")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=float(cfg['training']['lr']), 
                                  weight_decay=1e-4)

    # 5. Boucle d'entraînement
    best_loss = float('inf')
    epochs = cfg['training']['epochs']
    
    print(f"Début de l'entraînement pour {epochs} époques.")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Sécurité anti-crash si la liste est vide
        if len(train_loader) == 0:
            print("❌ Erreur: DataLoader vide.")
            break

        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Average Loss: {loss:.4f}")
        
        if loss < best_loss:
            best_loss = loss
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, "best_pretrained_model.pth")
            print("💾 Modèle sauvegardé !")

if __name__ == "__main__":
    main()