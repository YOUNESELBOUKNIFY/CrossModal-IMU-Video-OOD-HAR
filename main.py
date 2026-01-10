"""
Script principal pour exécuter le pipeline complet

Usage:
    python main.py --mode preprocess
    python main.py --mode pretrain
    python main.py --mode classify --classify-mode linear_probe
    python main.py --mode classify --classify-mode finetune
    python main.py --mode evaluate
    python main.py --mode all   # Tout le pipeline
"""

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import CONFIG
from src.data.preprocessing import MMEAPreprocessor
from src.data.datasets import create_dataloaders
from src.models.models import CrossModalModel, IMUClassifier
from src.models.losses import SigmoidContrastiveLoss
from src.train.trainer import CrossModalTrainer, ClassificationTrainer
from src.eval.evaluator import Evaluator, FewShotEvaluator
from src.utils import (
    set_seed, get_device, check_dataset_paths,
    print_model_info, plot_training_curves
)


class Pipeline:
    """
    Pipeline complet pour IMU-Video HAR
    """

    def __init__(self, config):
        self.config = config
        self.device = get_device()
        set_seed(config.training.seed)

        # Vérifier les chemins dataset
        if not check_dataset_paths(config):
            raise ValueError("Chemins du dataset invalides !")

        print("\n✓ Pipeline initialisé")
        print(f"Device: {self.device}")

    def run_preprocessing(self):
        """
        Étape 1: Preprocessing du dataset
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 1: PREPROCESSING")
        print("=" * 60)

        preprocessor = MMEAPreprocessor(self.config)
        results = preprocessor.run_full_preprocessing()

        # Afficher stats
        print("\n=== Statistiques ===")
        for split, df in results.items():
            print(f"{split}: {len(df)} windows, {df['class_name'].nunique()} classes")

        return results

    def run_pretraining(self):
        """
        Étape 2: Pretraining cross-modal
        Support 1 GPU ou multi-GPU via DataParallel.
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 2: CROSS-MODAL PRETRAINING")
        print("=" * 60)

        # Charger metadata
        train_df = pd.read_csv(self.config.paths.preprocessed_dir / "train_metadata.csv")
        val_df = pd.read_csv(self.config.paths.preprocessed_dir / "val_metadata.csv")
        test_df = pd.read_csv(self.config.paths.preprocessed_dir / "test_metadata.csv")

        # Dataloaders
        print("\nCréation des dataloaders...")
        dataloaders = create_dataloaders(
            self.config, train_df, val_df, test_df, mode="cross_modal"
        )

        # Modèle
        print("\nCréation du modèle...")
        model = CrossModalModel(self.config)
        print_model_info(model, "Cross-Modal Model")

        # Multi-GPU (DataParallel)
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if str(self.device).startswith("cuda") and n_gpu > 1:
            print(f"\n✓ Multi-GPU détecté: {n_gpu} GPUs -> DataParallel activé")
            model = torch.nn.DataParallel(model)
        else:
            print(f"\n✓ Single GPU/CPU: device={self.device}, n_gpu={n_gpu}")

        # Loss + Trainer
        loss_fn = SigmoidContrastiveLoss(learnable=True)
        trainer = CrossModalTrainer(model, loss_fn, self.config, device=str(self.device))

        # Fit
        print("\nDébut de l'entraînement...")
        trainer.fit(dataloaders["train"], dataloaders["val"])

        # Courbes
        plot_training_curves(
            trainer.history,
            save_path=self.config.paths.results_dir / "pretraining_curves.png",
        )

        print(f"\n✓ Pretraining terminé. Best val loss: {trainer.best_metric:.4f}")

        # Sauvegarde state_dict final (dé-wrappé si DataParallel)
        try:
            save_dir = self.config.paths.checkpoints_dir / "cross_modal"
            save_dir.mkdir(parents=True, exist_ok=True)
            final_path = save_dir / "final_model_state_dict.pt"

            state_dict = (
                trainer.model.module.state_dict()
                if isinstance(trainer.model, torch.nn.DataParallel)
                else trainer.model.state_dict()
            )
            torch.save(state_dict, final_path)
            print(f"✓ State_dict final sauvegardé: {final_path}")
        except Exception as e:
            print(f"Avertissement: sauvegarde state_dict final échouée: {e}")

        return trainer

    def run_classification(self, mode="both"):
        """
        Étape 3: Classification (linear probe + finetune)
        Args:
            mode: 'linear_probe', 'finetune', ou 'both'
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 3: CLASSIFICATION")
        print("=" * 60)

        # Charger metadata
        train_df = pd.read_csv(self.config.paths.preprocessed_dir / "train_metadata.csv")
        val_df = pd.read_csv(self.config.paths.preprocessed_dir / "val_metadata.csv")
        test_df = pd.read_csv(self.config.paths.preprocessed_dir / "test_metadata.csv")

        # Dataloaders classification
        clf_loaders = create_dataloaders(
            self.config, train_df, val_df, test_df, mode="classification"
        )

        # Charger checkpoint pretrain
        print("\nChargement de l'encodeur pré-entraîné...")
        checkpoint_path = self.config.paths.checkpoints_dir / "cross_modal" / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint introuvable: {checkpoint_path}\n"
                "Veuillez d'abord exécuter le pretraining !"
            )

        checkpoint = torch.load(checkpoint_path, map_location=str(self.device))

        cross_modal_model = CrossModalModel(self.config)
        # Important: si tu as sauvé un checkpoint depuis DataParallel, il peut avoir 'module.'.
        # Ici on suppose que ton Trainer gère déjà ça; sinon tu peux ajouter un petit strip.
        cross_modal_model.load_state_dict(checkpoint["model_state_dict"])
        pretrained_encoder = cross_modal_model.imu_encoder

        results = {}

        # Linear probing
        if mode in ["linear_probe", "both"]:
            print("\n--- Linear Probing ---")
            classifier_probe = IMUClassifier(
                pretrained_encoder, self.config, freeze_encoder=True
            )
            print_model_info(classifier_probe, "Classifier (Linear Probe)")

            trainer_probe = ClassificationTrainer(
                classifier_probe, self.config, device=str(self.device), mode="linear_probe"
            )
            trainer_probe.fit(clf_loaders["train"], clf_loaders["val"])

            evaluator_probe = Evaluator(classifier_probe, self.config, device=str(self.device))
            results["linear_probe"] = evaluator_probe.evaluate(clf_loaders["test"])

        # Full finetuning
        if mode in ["finetune", "both"]:
            print("\n--- Full Finetuning ---")
            classifier_ft = IMUClassifier(
                pretrained_encoder, self.config, freeze_encoder=False
            )
            print_model_info(classifier_ft, "Classifier (Finetune)")

            trainer_ft = ClassificationTrainer(
                classifier_ft, self.config, device=str(self.device), mode="finetune"
            )
            trainer_ft.fit(clf_loaders["train"], clf_loaders["val"])

            evaluator_ft = Evaluator(classifier_ft, self.config, device=str(self.device))
            results["finetune"] = evaluator_ft.evaluate(clf_loaders["test"])

        # Comparaison
        if mode == "both":
            print("\n=== Comparaison des résultats ===")
            comparison = pd.DataFrame(
                {
                    "Linear Probe": results["linear_probe"]["metrics"],
                    "Full Finetune": results["finetune"]["metrics"],
                }
            ).T
            print(comparison.to_string())

            comparison.to_csv(self.config.paths.results_dir / "classification_comparison.csv")

        return results

    def run_evaluation(self):
        """
        Étape 4: Évaluation complète (few-shot, ablations, etc.)
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 4: ÉVALUATION COMPLÈTE")
        print("=" * 60)

        train_df = pd.read_csv(self.config.paths.preprocessed_dir / "train_metadata.csv")
        test_df = pd.read_csv(self.config.paths.preprocessed_dir / "test_metadata.csv")

        checkpoint_path = self.config.paths.checkpoints_dir / "cross_modal" / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint introuvable: {checkpoint_path}\n"
                "Veuillez d'abord exécuter le pretraining !"
            )

        checkpoint = torch.load(checkpoint_path, map_location=str(self.device))
        cross_modal_model = CrossModalModel(self.config)
        cross_modal_model.load_state_dict(checkpoint["model_state_dict"])
        pretrained_encoder = cross_modal_model.imu_encoder

        # Few-shot
        print("\n--- Few-Shot Evaluation ---")
        fs_evaluator = FewShotEvaluator(self.config, device=str(self.device))

        fs_results = fs_evaluator.run_few_shot_experiments(
            pretrained_encoder,
            train_df,
            test_df,
            experiment_name="cross_modal_pretrained",
        )

        fs_agg = fs_evaluator.aggregate_results(fs_results)

        print("\n=== Few-Shot Results ===")
        print(fs_agg.to_string())

        fs_results.to_csv(self.config.paths.results_dir / "fewshot_results_raw.csv", index=False)
        fs_agg.to_csv(self.config.paths.results_dir / "fewshot_results_agg.csv", index=False)

        print("\n✓ Évaluation terminée")
        return fs_results, fs_agg

    def run_all(self):
        """
        Exécute le pipeline complet
        """
        print("\n" + "=" * 60)
        print("PIPELINE COMPLET - DÉBUT")
        print("=" * 60)

        # 1) Preprocess
        if not (self.config.paths.preprocessed_dir / "train_metadata.csv").exists():
            self.run_preprocessing()
        else:
            print("\n✓ Preprocessing déjà effectué")

        # 2) Pretrain
        if not (self.config.paths.checkpoints_dir / "cross_modal" / "best_model.pt").exists():
            self.run_pretraining()
        else:
            print("\n✓ Pretraining déjà effectué")

        # 3) Classification
        self.run_classification(mode="both")

        # 4) Eval
        self.run_evaluation()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLET - TERMINÉ")
        print("=" * 60)

        self.generate_final_report()

    def generate_final_report(self):
        """
        Génère un rapport final avec tous les résultats
        """
        import json
        from src.utils import create_results_summary  # <- important: chemin correct

        print("\n=== Génération du rapport final ===")

        results_dir = self.config.paths.results_dir
        report = {
            "config": {
                "seed": self.config.training.seed,
                "pretrain_epochs": self.config.training.pretrain_epochs,
                "train_epochs": self.config.training.train_epochs,
            },
            "files": create_results_summary(results_dir),
        }

        # Charger résultats si dispo
        try:
            classification_results = pd.read_csv(
                results_dir / "classification_comparison.csv", index_col=0
            )
            report["classification"] = classification_results.to_dict()
        except Exception:
            pass

        try:
            fewshot_results = pd.read_csv(results_dir / "fewshot_results_agg.csv")
            report["fewshot_summary"] = {
                "samples": fewshot_results["n_samples"].unique().tolist(),
                "modes": fewshot_results["mode"].unique().tolist(),
            }
        except Exception:
            pass

        report_path = results_dir / "final_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n✓ Rapport final sauvegardé: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline IMU-Video Cross-modal HAR")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["preprocess", "pretrain", "classify", "evaluate", "all"],
        default="all",
        help="Mode d'exécution",
    )
    parser.add_argument(
        "--classify-mode",
        type=str,
        choices=["linear_probe", "finetune", "both"],
        default="both",
        help="Mode de classification",
    )

    args = parser.parse_args()

    pipeline = Pipeline(CONFIG)

    if args.mode == "preprocess":
        pipeline.run_preprocessing()
    elif args.mode == "pretrain":
        pipeline.run_pretraining()
    elif args.mode == "classify":
        pipeline.run_classification(mode=args.classify_mode)
    elif args.mode == "evaluate":
        pipeline.run_evaluation()
    elif args.mode == "all":
        pipeline.run_all()

    print("\n✓ Terminé!")


if __name__ == "__main__":
    main()
