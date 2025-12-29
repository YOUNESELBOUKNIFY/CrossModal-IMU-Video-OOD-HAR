import torch
import torch.nn as nn  # Nécessaire pour DataParallel
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm

# Assurez-vous que les imports correspondent à votre structure de dossiers
from src.models.contrastive import CrossModalContrastiveModel
from src.data.dataset import MMEADataset 

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    # La barre de progression
    loop = tqdm(dataloader, desc="Training")
    
    for batch_idx, (imu, video) in enumerate(loop):
        # Envoi des données sur le GPU principal (DataParallel s'occupe de la distribution ensuite)
        imu = imu.to(device)
        video = video.to(device)
        
        optimizer.zero_grad()
        
        # --- Forward pass ---
        # Avec DataParallel, imu et video sont coupés en 2 (un morceau par GPU).
        # Le modèle s'exécute en parallèle.
        # Les résultats (imu_feat, video_feat) sont rassemblés sur le GPU 0.
        imu_feat, video_feat = model(imu, video)
        
        # --- Calcul de la Loss ---
        # ATTENTION : DataParallel cache vos méthodes perso dans 'module'.
        # Il faut vérifier si le modèle est enveloppé ou non.
        if isinstance(model, nn.DataParallel):
            loss = model.module.compute_loss(imu_feat, video_feat)
        else:
            loss = model.compute_loss(imu_feat, video_feat)
        
        # --- Backward ---
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def main():
    # 1. Chargement de la config
    cfg = load_config('configs/config.yaml')
    
    # 2. Configuration des GPUs pour Kaggle
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = torch.device('cuda')
        print(f"✅ {num_gpus} GPU(s) détecté(s) !")
        if num_gpus > 1:
            print(f"   Les GPUs seront utilisés en parallèle : {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    else:
        device = torch.device('cpu')
        print("⚠️ Aucun GPU détecté. L'entraînement sera très lent.")

    # 3. Préparation des données
    # (Remplacez [] par votre logique de chargement de liste de fichiers)
    train_dataset = MMEADataset([], mode='pretrain', 
                                imu_params=cfg['preprocessing']['imu'],
                                video_params=cfg['preprocessing']['video'])
    
    # Optimisation pour Kaggle : num_workers=4 utilise les cœurs CPU pour charger vite
    # pin_memory=True accélère le transfert RAM -> VRAM
    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg['training']['batch_size'], 
                              shuffle=True, 
                              num_workers=4,  
                              pin_memory=True)

    # 4. Initialisation du modèle
    print("Construction du modèle...")
    model = CrossModalContrastiveModel(cfg).to(device)
    
    # --- ACTIVATION MULTI-GPU ---
    if torch.cuda.device_count() > 1:
        print("🚀 Activation de DataParallel pour utiliser les 2 GPUs...")
        model = nn.DataParallel(model)
    # ----------------------------

    # Optimiseur
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=float(cfg['training']['lr']), 
                                  weight_decay=1e-4)

    # 5. Boucle d'entraînement
    best_loss = float('inf')
    epochs = cfg['training']['epochs']
    
    print(f"Début de l'entraînement pour {epochs} époques.")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Average Loss: {loss:.4f}")
        
        # Sauvegarde du meilleur modèle
        if loss < best_loss:
            best_loss = loss
            
            # ASTUCE : Sauvegarder 'model.module' si DataParallel est utilisé
            # Cela permet de recharger le modèle plus tard sur 1 seul GPU sans erreur de clés
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            torch.save(state_dict, "best_pretrained_model.pth")
            print("💾 Modèle sauvegardé (format standard) !")

if __name__ == "__main__":
    main()