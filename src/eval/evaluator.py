"""
Évaluation des modèles et génération de tableaux comparatifs
"""
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, 
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class Evaluator:
    """Classe pour évaluer les modèles et générer des rapports"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, dataloader):
        """
        Génère les prédictions sur un dataloader
        Returns:
            predictions: np.array
            labels: np.array
            logits: np.array
        """
        all_preds = []
        all_labels = []
        all_logits = []
        
        for batch in tqdm(dataloader, desc="Prédiction"):
            imu = batch['imu'].to(self.device)
            labels = batch['label']
            
            logits = self.model(imu)
            _, preds = logits.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_logits.append(logits.cpu().numpy())
        
        return (np.array(all_preds), 
                np.array(all_labels), 
                np.vstack(all_logits))
    
    def compute_metrics(self, y_true, y_pred):
        """Calcule toutes les métriques"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred) * 100,
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred) * 100,
            'f1_macro': f1_score(y_true, y_pred, average='macro') * 100,
            'f1_weighted': f1_score(y_true, y_pred, average='weighted') * 100,
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        }
        return metrics
    
    def evaluate(self, dataloader):
        """Évaluation complète"""
        preds, labels, logits = self.predict(dataloader)
        metrics = self.compute_metrics(labels, preds)
        
        return {
            'metrics': metrics,
            'predictions': preds,
            'labels': labels,
            'logits': logits
        }
    
    def generate_classification_report(self, y_true, y_pred, class_names=None):
        """Génère un rapport de classification détaillé"""
        if class_names is None:
            class_names = [str(i) for i in range(self.config.model.num_classes)]
        
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        return pd.DataFrame(report).transpose()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = [str(i) for i in range(self.config.model.num_classes)]
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix sauvegardée: {save_path}")
        
        plt.close()


class FewShotEvaluator:
    """
    Évaluateur pour expériences few-shot
    Réplique les évaluations de l'article (Table 3)
    """
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
    
    def run_few_shot_experiments(self, 
                                  pretrained_encoder,
                                  train_df,
                                  test_df,
                                  experiment_name="default"):
        """
        Lance les expériences few-shot avec différents nombres de samples
        
        Returns:
            DataFrame avec résultats
        """
        from datasets import FewShotSampler, IMUClassificationDataset, DataLoader
        from models import IMUClassifier
        from trainer import ClassificationTrainer
        
        results = []
        
        # Pour chaque nombre de samples
        for n_samples in self.config.eval.few_shot_samples:
            print(f"\n{'='*60}")
            print(f"Few-shot avec {n_samples} samples par classe")
            print(f"{'='*60}")
            
            # Répéter plusieurs fois avec différents seeds
            for run in range(self.config.eval.few_shot_runs):
                print(f"\nRun {run + 1}/{self.config.eval.few_shot_runs}")
                
                # Sample subset
                sampler = FewShotSampler(train_df, self.config)
                train_subset = sampler.sample_k_per_class(n_samples, seed=run + 42)
                
                # Create dataloaders
                train_dataset = IMUClassificationDataset(train_subset, self.config)
                test_dataset = IMUClassificationDataset(test_df, self.config)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
                
                # Linear probing
                print("  Linear probing...")
                model_probe = IMUClassifier(
                    pretrained_encoder,
                    self.config,
                    freeze_encoder=True
                )
                
                trainer_probe = ClassificationTrainer(
                    model_probe, self.config, self.device, mode='linear_probe'
                )
                trainer_probe.fit(train_loader, test_loader)
                
                # Evaluate
                evaluator_probe = Evaluator(model_probe, self.config, self.device)
                eval_probe = evaluator_probe.evaluate(test_loader)
                
                # Full finetuning
                print("  Full finetuning...")
                model_ft = IMUClassifier(
                    pretrained_encoder,
                    self.config,
                    freeze_encoder=False
                )
                
                trainer_ft = ClassificationTrainer(
                    model_ft, self.config, self.device, mode='finetune'
                )
                trainer_ft.fit(train_loader, test_loader)
                
                evaluator_ft = Evaluator(model_ft, self.config, self.device)
                eval_ft = evaluator_ft.evaluate(test_loader)
                
                # Store results
                results.append({
                    'experiment': experiment_name,
                    'n_samples': n_samples,
                    'run': run,
                    'mode': 'linear_probe',
                    **eval_probe['metrics']
                })
                
                results.append({
                    'experiment': experiment_name,
                    'n_samples': n_samples,
                    'run': run,
                    'mode': 'finetune',
                    **eval_ft['metrics']
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def aggregate_results(self, results_df):
        """
        Agrège les résultats (moyenne ± std) comme dans l'article
        """
        agg_df = results_df.groupby(['experiment', 'n_samples', 'mode']).agg({
            'balanced_accuracy': ['mean', 'std'],
            'f1_macro': ['mean', 'std'],
            'accuracy': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
        
        return agg_df
    
    def create_comparison_table(self, results_dict):
        """
        Crée un tableau de comparaison style article (Table 3)
        
        Args:
            results_dict: dict avec clés = nom expérience, valeurs = results_df
        """
        all_results = []
        
        for exp_name, df in results_dict.items():
            agg = self.aggregate_results(df)
            agg['experiment'] = exp_name
            all_results.append(agg)
        
        comparison_df = pd.concat(all_results, ignore_index=True)
        
        # Format pour affichage
        comparison_df['balanced_accuracy_formatted'] = comparison_df.apply(
            lambda row: f"{row['balanced_accuracy_mean']:.2f} ± {row['balanced_accuracy_std']:.2f}",
            axis=1
        )
        
        # Pivot pour avoir format article
        pivot = comparison_df.pivot_table(
            index=['experiment', 'mode'],
            columns='n_samples',
            values='balanced_accuracy_formatted',
            aggfunc='first'
        )
        
        return pivot


def compare_baseline_vs_pretrained(config, test_df):
    """
    Compare baseline (from scratch) vs pretrained encoder
    """
    from models import IMUEncoder, IMUClassifier
    from trainer import ClassificationTrainer
    from datasets import IMUClassificationDataset
    from torch.utils.data import DataLoader
    
    results = {}
    
    # 1) Baseline: train from scratch
    print("\n=== Training Baseline (from scratch) ===")
    baseline_encoder = IMUEncoder(config)
    baseline_model = IMUClassifier(baseline_encoder, config, freeze_encoder=False)
    
    test_dataset = IMUClassificationDataset(test_df, config)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Suppose training déjà fait, charger checkpoint
    # baseline_trainer.fit(train_loader, val_loader)
    
    evaluator = Evaluator(baseline_model, config)
    baseline_results = evaluator.evaluate(test_loader)
    results['baseline'] = baseline_results['metrics']
    
    # 2) Pretrained: load pretrained encoder
    print("\n=== Evaluating Pretrained Encoder ===")
    # Load checkpoint
    # pretrained_encoder.load_state_dict(...)
    
    # pretrained_model = IMUClassifier(pretrained_encoder, config)
    # pretrained_results = evaluator.evaluate(test_loader)
    # results['pretrained'] = pretrained_results['metrics']
    
    # Create comparison table
    comparison_df = pd.DataFrame(results).T
    return comparison_df


def save_results_table(df, save_path, format='csv'):
    """Sauvegarde un DataFrame de résultats"""
    save_path = Path(save_path)
    
    if format == 'csv':
        df.to_csv(save_path)
    elif format == 'latex':
        df.to_latex(save_path)
    elif format == 'markdown':
        df.to_markdown(save_path)
    
    print(f"Résultats sauvegardés: {save_path}")


if __name__ == "__main__":
    print("Module evaluator.py - Prêt pour évaluation")