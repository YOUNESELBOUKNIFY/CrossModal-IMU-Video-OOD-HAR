import sys
import os

# --- CORRECTION 1 : GESTION DES CHEMINS (IMPÉRATIF) ---
# Ajoute la racine du projet au path Python pour que 'from src...' fonctionne
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# -----------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

# Imports du projet
from src.models.contrastive import CrossModalContrastiveModel
from src.data.dataset import MMEADataset 

def load_config(config_path):
    # Utilisation du chemin absolu pour éviter les erreurs
    full_path = os.path.join(project_root, config_path)
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)

# Ajoutez cet import en haut si pas déjà présent (normalement torch l'a déjà)
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    # Initialisation du Scaler pour la précision mixte
    scaler = GradScaler() 
    
    loop = tqdm(dataloader, desc="Training")
    
    for batch_idx, (imu, video) in enumerate(loop):
        imu = imu.to(device)
        video = video.to(device)
        
        optimizer.zero_grad()
        
        # --- Optimisation Mémoire (AMP) ---
        # On utilise autocast pour que le GPU utilise moins de RAM (FP16)
        with autocast():
            imu_feat, video_feat = model(imu, video)
            
            if isinstance(model, nn.DataParallel):
                loss = model.module.compute_loss(imu_feat, video_feat)
            else:
                loss = model.compute_loss(imu_feat, video_feat)
        
        # Backward pass avec le scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # ----------------------------------
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
        # Optionnel : Vider le cache GPU si on est vraiment limite
        torch.cuda.empty_cache() 
        
    return total_loss / len(dataloader)

def main():
    # 1. Chargement de la config
    print("⚙️ Chargement de la configuration...")
    cfg = load_config('configs/config.yaml')
    
    # 2. Configuration des GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device('cuda')
        print(f"✅ {num_gpus} GPU(s) détecté(s) !")
        if num_gpus > 1:
            print(f"   Mode Multi-GPU activé.")
    else:
        device = torch.device('cpu')
        print("⚠️ Aucun GPU détecté.")

    # 3. Préparation des données (CORRECTION MAJEURE ICI)
    print("📂 Lecture et préparation des données...")
    
    # Chemin racine vers vos données Kaggle (Input)
    dataset_root = "/kaggle/input/dataset-har/UESTC-MMEA-CL"
    train_txt_path = os.path.join(dataset_root, "train.txt")
    
    # Vérification de l'existence du fichier train.txt
    if not os.path.exists(train_txt_path):
        print(f"❌ ERREUR CRITIQUE : Le fichier {train_txt_path} n'existe pas.")
        return

    formatted_data = []
    
    # Lecture ligne par ligne et construction des chemins complets
    with open(train_txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            filename = line.strip() # ex: "1_10_2"
            if not filename: continue
            
            # --- Construction des chemins ---
            # NOTE : Vérifiez si vos fichiers sont .csv et .mp4 ou .avi
            p_imu = os.path.join(dataset_root, 'sensor', filename + ".csv")
            p_video = os.path.join(dataset_root, 'video', filename + ".mp4") # ou .avi ?
            
            # --- Extraction du Label ---
            # Format attendu : Sujet_Action_Repetition (ex: 1_10_2)
            try:
                parts = filename.split('_')
                # Action est à l'index 1. On soustrait 1 pour avoir un index base-0 (0 à N-1)
                label = int(parts[1]) - 1 
            except:
                label = 0 # Valeur par défaut en cas de nom bizarre
            
            # On ajoute le tuple (imu, video, label)
            formatted_data.append((p_imu, p_video, label))

    print(f"✅ {len(formatted_data)} échantillons chargés avec succès !")

    # Vérification rapide avant de lancer
    if len(formatted_data) > 0:
        if not os.path.exists(formatted_data[0][0]):
            print(f"⚠️ ALERTE : Le fichier {formatted_data[0][0]} est introuvable.")
            print("   >> Vérifiez les extensions (.csv ? .txt ?) dans le dossier 'sensor'.")

    # Création du Dataset avec la liste formatée
    train_dataset = MMEADataset(formatted_data, mode='pretrain', 
                                imu_params=cfg['preprocessing']['imu'],
                                video_params=cfg['preprocessing']['video'])
    
    # Création du DataLoader
    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg['training']['batch_size'], 
                              shuffle=True, 
                              num_workers=4,  
                              pin_memory=True)

    # 4. Initialisation du modèle
    print("🏗️ Construction du modèle...")
    model = CrossModalContrastiveModel(cfg).to(device)
    
    # Activation DataParallel pour multi-GPU
    if torch.cuda.device_count() > 1:
        print("🚀 Activation de DataParallel...")
        model = nn.DataParallel(model)

    # Optimiseur
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=float(cfg['training']['lr']), 
                                  weight_decay=1e-4)

    # 5. Boucle d'entraînement
    best_loss = float('inf')
    epochs = cfg['training']['epochs']
    
    print(f"🏁 Début de l'entraînement pour {epochs} époques.")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        if len(train_loader) == 0:
            print("❌ Erreur : DataLoader vide.")
            break

        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Average Loss: {loss:.4f}")
        
        # Sauvegarde
        if loss < best_loss:
            best_loss = loss
            # Gestion de la sauvegarde DataParallel vs Single GPU
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, "best_pretrained_model.pth")
            print("💾 Modèle sauvegardé !")

if __name__ == "__main__":
    main()