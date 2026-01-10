"""
Script principal pour exécuter le pipeline complet IMU-Video HAR

Usage:
    python main.py --mode preprocess
    python main.py --mode pretrain
    python main.py --mode classify
    python main.py --mode evaluate
    python main.py --mode all
"""

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd

# Ajouter la racine du projet au PYTHONPATH
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

from configs.config import CONFIG
from src.data.preprocessing import MMEAPreprocessor
from src.data.datasets import create_dataloaders
from src.models.models import CrossModalModel, IMUClassifier
from src.models.losses import SigmoidContrastiveLoss
from src.train.trainer import CrossModalTrainer, ClassificationTrainer
from src.eval.evaluator import Evaluator, FewShotEvaluator
from src.utils import (
    set_seed,
    get_device,
    check_dataset_paths,
    print_model_info,
    plot_training_curves,
)


# ======================================================================
# PIPELINE
# ======================================================================
class Pipeline:
    """Pipeline complet IMU–Vidéo pour HAR"""

    def __init__(self, config):
        self.config = config
        self.device = get_device()
        set_seed(config.training.seed)

        if not check_dataset_paths(config):
            raise ValueError("Chemins du dataset invalides")

        print("\n✓ Pipeline initialisé")
        print(f"✓ Device utilisé : {self.device}")

    # ------------------------------------------------------------------
    # ÉTAPE 1 : PREPROCESSING
    # ------------------------------------------------------------------
    def run_preprocessing(self):
        print("\n" + "=" * 60)
        print("ÉTAPE 1 : PREPROCESSING")
        print("=" * 60)

        preprocessor = MMEAPreprocessor(self.config)
        results = preprocessor.run_full_preprocessing()

        print("\n=== Statistiques ===")
        for split, df in results.items():
            print(
                f"{split}: {len(df)} windows | "
                f"{df['class_name'].nunique()} classes"
            )

        return results

    # ------------------------------------------------------------------
    # ÉTAPE 2 : PRETRAINING CROSS-MODAL
    # ------------------------------------------------------------------
    def run_pretraining(self):
        """
        Pretraining cross-modal IMU–Vidéo
        Support mono-GPU et multi-GPU (DataParallel)
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 2 : CROSS-MODAL PRETRAINING")
        print("=" * 60)

        # Charger metadata
        train_df = pd.read_csv(self.config.paths.preprocessed_dir / "train_metadata.csv")
        val_df = pd.read_csv(self.config.paths.preprocessed_dir / "val_metadata.csv")
        test_df = pd.read_csv(self.config.paths.preprocessed_dir / "test_metadata.csv")

        print("\nCréation des dataloaders...")
        dataloaders = create_dataloaders(
            self.config, train_df, val_df, test_df, mode="cross_modal"
        )

        print("\nCréation du modèle...")
        model = CrossModalModel(self.config)
        print_model_info(model, "Cross-Modal Model")

        # Multi-GPU
        n_gpu = torch.cuda.device_count()
        if self.device.startswith("cuda") and n_gpu > 1:
            print(f"\n✓ Multi-GPU détecté ({n_gpu}) → DataParallel activé")
            model = torch.nn.DataParallel(model)
        else:
            print(f"\n✓ Single GPU / CPU (n_gpu={n_gpu})")

        loss_fn = SigmoidContrastiveLoss(learnable=True)
        trainer = CrossModalTrainer(model, loss_fn, self.config, self.device)

        print("\nDébut de l'entraînement...")
        trainer.fit(dataloaders["train"], dataloaders["val"])

        plot_training_curves(
            trainer.history,
            save_path=self.config.paths.results_dir / "pretraining_curves.png",
        )

        print(f"\n✓ Pretraining terminé | Best val loss = {trainer.best_metric:.4f}")

        # Sauvegarde state_dict final
        save_dir = self.config.paths.checkpoints_dir / "cross_modal"
        save_dir.mkdir(parents=True, exist_ok=True)

        state_dict = (
            trainer.model.module.state_dict()
            if isinstance(trainer.model, torch.nn.DataParallel)
            else trainer.model.state_dict()
        )

        torch.save(state_dict, save_dir / "final_model_state_dict.pt")
        print("✓ State_dict final sauvegardé")

        return trainer

    # ------------------------------------------------------------------
    # ÉTAPE 3 : CLASSIFICATION
    # ------------------------------------------------------------------
    def run_classification(self, mode="both"):
        print("\n" + "=" * 60)
        print("ÉTAPE 3 : CLASSIFICATION")
        print("=" * 60)

        train_df = pd.read_csv(self.config.paths.preprocessed_dir / "train_metadata.csv")
        val_df = pd.read_csv(self.config.paths.preprocessed_dir / "val_metadata.csv")
        test_df = pd.read_csv(self.config.paths.preprocessed_dir / "test_metadata.csv")

        loaders = create_dataloaders(
            self.config, train_df, val_df, test_df, mode="classification"
        )

        checkpoint_path = (
            self.config.paths.checkpoints_dir / "cross_modal" / "best_model.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError("Veuillez exécuter le pretraining d'abord.")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = CrossModalModel(self.config)
        model.load_state_dict(checkpoint["model_state_dict"])
        pretrained_encoder = model.imu_encoder

        results = {}

        if mode in ["linear_probe", "both"]:
            print("\n--- Linear Probing ---")
            clf = IMUClassifier(
                pretrained_encoder, self.config, freeze_encoder=True
            )
            trainer = ClassificationTrainer(
                clf, self.config, self.device, mode="linear_probe"
            )
            trainer.fit(loaders["train"], loaders["val"])

            evaluator = Evaluator(clf, self.config, self.device)
            results["linear_probe"] = evaluator.evaluate(loaders["test"])

        if mode in ["finetune", "both"]:
            print("\n--- Full Finetuning ---")
            clf = IMUClassifier(
                pretrained_encoder, self.config, freeze_encoder=False
            )
            trainer = ClassificationTrainer(
                clf, self.config, self.device, mode="finetune"
            )
            trainer.fit(loaders["train"], loaders["val"])

            evaluator = Evaluator(clf, self.config, self.device)
            results["finetune"] = evaluator.evaluate(loaders["test"])

        return results

    # ------------------------------------------------------------------
    # ÉTAPE 4 : ÉVALUATION
    # ------------------------------------------------------------------
    def run_evaluation(self):
        print("\n" + "=" * 60)
        print("ÉTAPE 4 : ÉVALUATION COMPLÈTE")
        print("=" * 60)

        train_df = pd.read_csv(self.config.paths.preprocessed_dir / "train_metadata.csv")
        test_df = pd.read_csv(self.config.paths.preprocessed_dir / "test_metadata.csv")

        checkpoint_path = (
            self.config.paths.checkpoints_dir / "cross_modal" / "best_model.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model = CrossModalModel(self.config)
        model.load_state_dict(checkpoint["model_state_dict"])
        encoder = model.imu_encoder

        fs_evaluator = FewShotEvaluator(self.config, self.device)
        fs_results = fs_evaluator.run_few_shot_experiments(
            encoder, train_df, test_df, "cross_modal_pretrained"
        )

        fs_agg = fs_evaluator.aggregate_results(fs_results)
        print(fs_agg)

        return fs_results, fs_agg

    # ------------------------------------------------------------------
    # PIPELINE COMPLET
    # ------------------------------------------------------------------
    def run_all(self):
        self.run_preprocessing()
        self.run_pretraining()
        self.run_classification(mode="both")
        self.run_evaluation()


# ======================================================================
# MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline IMU-Video Cross-modal HAR"
    )

    parser.add_argument(
        "--mode",
        choices=["preprocess", "pretrain", "classify", "evaluate", "all"],
        default="all",
    )
    parser.add_argument(
        "--classify-mode",
        choices=["linear_probe", "finetune", "both"],
        default="both",
    )

    args = parser.parse_args()
    pipeline = Pipeline(CONFIG)

    if args.mode == "preprocess":
        pipeline.run_preprocessing()
    elif args.mode == "pretrain":
        pipeline.run_pretraining()
    elif args.mode == "classify":
        pipeline.run_classification(args.classify_mode)
    elif args.mode == "evaluate":
        pipeline.run_evaluation()
    else:
        pipeline.run_all()

    print("\n✓ Pipeline terminé")


if __name__ == "__main__":
    main()
