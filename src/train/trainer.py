"""
Trainer classes pour pretraining et classification
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json


class BaseTrainer:
    """Trainer de base avec fonctionnalités communes"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.history = {'train': [], 'val': []}
        
    def save_checkpoint(self, path, **kwargs):
        """Sauvegarde un checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_metric': self.best_metric,
            'history': self.history,
            **kwargs
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint sauvegardé: {path}")
    
    def load_checkpoint(self, path):
        """Charge un checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        self.history = checkpoint.get('history', {'train': [], 'val': []})
        print(f"Checkpoint chargé: {path}")
        return checkpoint


class CrossModalTrainer(BaseTrainer):
    """
    Trainer pour pretraining cross-modal IMU-Video
    """
    
    def __init__(self, model, loss_fn, config, device='cuda'):
        super().__init__(model, config, device)
        self.loss_fn = loss_fn
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.pretrain_lr,
            weight_decay=config.training.pretrain_weight_decay
        )
        
        # Scheduler avec warmup
        num_epochs = config.training.pretrain_epochs
        warmup_epochs = config.training.pretrain_warmup_epochs
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-6
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    
    def train_epoch(self, dataloader):
        """Entraîne une epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            imu = batch['imu'].to(self.device)
            video = batch['video'].to(self.device)
            
            # Forward
            imu_proj, video_proj = self.model(imu, video)
            
            # Loss
            loss = self.loss_fn(imu_proj, video_proj)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, dataloader):
        """Validation"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Validation"):
            imu = batch['imu'].to(self.device)
            video = batch['video'].to(self.device)
            
            imu_proj, video_proj = self.model(imu, video)
            loss = self.loss_fn(imu_proj, video_proj)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def fit(self, train_loader, val_loader):
        """Entraînement complet"""
        num_epochs = self.config.training.pretrain_epochs
        save_dir = self.config.paths.checkpoints_dir / 'cross_modal'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        patience_counter = 0
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train'].append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.history['val'].append(val_loss)
            
            # Scheduler step
            self.scheduler.step()
            
            # Logging
            print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Checkpointing
            if val_loss < self.best_metric:
                self.best_metric = val_loss
                patience_counter = 0
                
                if self.config.training.save_best_only:
                    save_path = save_dir / 'best_model.pt'
                    self.save_checkpoint(
                        save_path,
                        optimizer_state_dict=self.optimizer.state_dict()
                    )
            else:
                patience_counter += 1
            
            # Save périodique
            if (epoch + 1) % self.config.training.save_every == 0:
                save_path = save_dir / f'checkpoint_epoch_{epoch}.pt'
                self.save_checkpoint(save_path)
            
            # Early stopping
            if patience_counter >= self.config.training.patience:
                print(f"Early stopping à l'epoch {epoch}")
                break
        
        # Sauvegarder l'historique
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


class ClassificationTrainer(BaseTrainer):
    """
    Trainer pour classification d'activités
    Supporte linear probing et full finetuning
    """
    
    def __init__(self, model, config, device='cuda', mode='linear_probe'):
        """
        Args:
            mode: 'linear_probe' ou 'finetune'
        """
        super().__init__(model, config, device)
        self.mode = mode
        
        # Loss
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Optimizer
        if mode == 'linear_probe':
            # Freeze encoder, optimiser seulement le classifier
            for param in model.imu_encoder.parameters():
                param.requires_grad = False
            
            self.optimizer = AdamW(
                model.classifier.parameters(),
                lr=config.training.train_lr_head,
                weight_decay=config.training.pretrain_weight_decay
            )
        
        elif mode == 'finetune':
            # Unfreeze encoder
            model.unfreeze_encoder()
            
            # Different LR pour encoder et classifier
            self.optimizer = AdamW([
                {'params': model.imu_encoder.parameters(), 
                 'lr': config.training.train_lr_encoder},
                {'params': model.classifier.parameters(), 
                 'lr': config.training.train_lr_head}
            ], weight_decay=config.training.pretrain_weight_decay)
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.train_epochs,
            eta_min=1e-7
        )
    
    def train_epoch(self, dataloader):
        """Entraîne une epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            imu = batch['imu'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward
            logits = self.model(imu)
            loss = self.loss_fn(logits, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    @torch.no_grad()
    def validate(self, dataloader):
        """Validation avec métriques détaillées"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        for batch in tqdm(dataloader, desc="Validation"):
            imu = batch['imu'].to(self.device)
            labels = batch['label'].to(self.device)
            
            logits = self.model(imu)
            loss = self.loss_fn(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        # Balanced accuracy
        from sklearn.metrics import balanced_accuracy_score, f1_score
        balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100
        f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_macro': f1_macro
        }
    
    def fit(self, train_loader, val_loader):
        """Entraînement complet"""
        num_epochs = self.config.training.train_epochs
        save_dir = self.config.paths.checkpoints_dir / f'classifier_{self.mode}'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        patience_counter = 0
        best_val_acc = 0
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history['train'].append(train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.history['val'].append(val_metrics)
            
            # Scheduler
            self.scheduler.step()
            
            # Logging
            print(f"\nEpoch {epoch}:")
            print(f"  Train - loss: {train_metrics['loss']:.4f}, acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val   - loss: {val_metrics['loss']:.4f}, " 
                  f"acc: {val_metrics['accuracy']:.2f}%, "
                  f"balanced_acc: {val_metrics['balanced_accuracy']:.2f}%, "
                  f"f1: {val_metrics['f1_macro']:.2f}%")
            
            # Checkpointing (based on balanced accuracy)
            if val_metrics['balanced_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['balanced_accuracy']
                patience_counter = 0
                
                save_path = save_dir / 'best_model.pt'
                self.save_checkpoint(save_path)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.training.patience:
                print(f"Early stopping à l'epoch {epoch}")
                break
        
        # Sauvegarder historique
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return best_val_acc