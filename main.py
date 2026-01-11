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

        if not check_dataset_paths(config):
            raise ValueError("Chemins du dataset invalides !")

        print("\n✓ Pipeline initialisé")
        print(f"Device: {self.device}")

    def run_preprocessing(self):
        print("\n" + "=" * 60)
        print("ÉTAPE 1: PREPROCESSING")
        print("=" * 60)

        preprocessor = MMEAPreprocessor(self.config)
        results = preprocessor.run_full_preprocessing()

        print("\n=== Statistiques ===")
        for split, df in results.items():
            print(f"{split}: {len(df)} windows, {df['class_name'].nunique()} classes")

        return results

    def _sample_k_per_class(self, df: pd.DataFrame, k: int, seed: int = 42) -> pd.DataFrame:
        """
        Prend k échantillons par classe (colonne 'label').
        - k <= 0 => retourne df complet
        - si une classe a < k => on garde tout
        """
        if k is None or int(k) <= 0:
            return df

        if "label" not in df.columns:
            raise ValueError("La colonne 'label' est introuvable dans metadata_df.")

        k = int(k)

        grouped = df.groupby("label", group_keys=False)

        def _sample(g):
            if len(g) <= k:
                return g
            return g.sample(n=k, random_state=seed)

        out = grouped.apply(_sample).reset_index(drop=True)
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
        return out

    def run_pretraining(self):
        """
        Étape 2: Pretraining cross-modal
        Support 1 GPU ou multi-GPU via DataParallel.
        + Option: utiliser un subset K par classe (pour accélérer)
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 2: CROSS-MODAL PRETRAINING")
        print("=" * 60)

        train_df = pd.read_csv(self.config.paths.preprocessed_dir / "train_metadata.csv")
        val_df   = pd.read_csv(self.config.paths.preprocessed_dir / "val_metadata.csv")
        test_df  = pd.read_csv(self.config.paths.preprocessed_dir / "test_metadata.csv")

        # ---------- Subset K / classe (optionnel) ----------
        tcfg = self.config.training
        use_subset = bool(getattr(tcfg, "use_subset", False))

        if use_subset:
            seed = int(getattr(tcfg, "subset_seed", 42))
            k_train = int(getattr(tcfg, "k_per_class_train", 0))
            k_val   = int(getattr(tcfg, "k_per_class_val", 0))
            k_test  = int(getattr(tcfg, "k_per_class_test", 0))

            if k_train > 0:
                train_df = self._sample_k_per_class(train_df, k_train, seed)
            if k_val > 0:
                val_df = self._sample_k_per_class(val_df, k_val, seed)
            if k_test > 0:
                test_df = self._sample_k_per_class(test_df, k_test, seed)

            print("\n[Subset K/class activé]")
            print(f"  Train: {len(train_df)} rows | classes={train_df['label'].nunique()} | k={k_train}")
            print(f"  Val  : {len(val_df)} rows | classes={val_df['label'].nunique()} | k={k_val}")
            print(f"  Test : {len(test_df)} rows | classes={test_df['label'].nunique()} | k={k_test}")
        else:
            print("\n[Subset K/class désactivé] -> dataset complet utilisé")

        # ---------- Dataloaders ----------
        print("\nCréation des dataloaders...")
        dataloaders = create_dataloaders(
            self.config, train_df, val_df, test_df, mode="cross_modal"
        )

        # ---------- Modèle ----------
        print("\nCréation du modèle...")
        model = CrossModalModel(self.config)
        print_model_info(model, "Cross-Modal Model")

        # ---------- Multi-GPU (DataParallel) ----------
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if str(self.device).startswith("cuda") and n_gpu > 1:
            print(f"\n✓ Multi-GPU détecté: {n_gpu} GPUs -> DataParallel activé")
            model = torch.nn.DataParallel(model)
        else:
            print(f"\n✓ Single GPU/CPU: device={self.device}, n_gpu={n_gpu}")

        # ---------- Trainer ----------
        loss_fn = SigmoidContrastiveLoss(learnable=True)
        trainer = CrossModalTrainer(model, loss_fn, self.config, device=str(self.device))

        print("\nDébut de l'entraînement...")
        trainer.fit(dataloaders["train"], dataloaders["val"])

        plot_training_curves(
            trainer.history,
            save_path=self.config.paths.results_dir / "pretraining_curves.png",
        )

        best_val = getattr(trainer, "best_metric", None)
        if best_val is None:
            best_val = getattr(trainer, "best_val_loss", None)

        print(f"\n✓ Pretraining terminé. Best val: {best_val}")
        return trainer
    def run_classification(self, mode="both"):
        """
        Étape 3: Classification (linear probe + finetune)
        """
        import copy

        print("\n" + "=" * 60)
        print("ÉTAPE 3: CLASSIFICATION")
        print("=" * 60)

        train_df = pd.read_csv(self.config.paths.preprocessed_dir / "train_metadata.csv")
        val_df   = pd.read_csv(self.config.paths.preprocessed_dir / "val_metadata.csv")
        test_df  = pd.read_csv(self.config.paths.preprocessed_dir / "test_metadata.csv")

        clf_loaders = create_dataloaders(
            self.config, train_df, val_df, test_df, mode="classification"
        )

        print("\nChargement de l'encodeur pré-entraîné...")
        checkpoint_path = self.config.paths.checkpoints_dir / "cross_modal" / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint introuvable: {checkpoint_path}\n"
                "Veuillez d'abord exécuter le pretraining !"
            )

        checkpoint = torch.load(checkpoint_path, map_location=str(self.device))

        cross_modal_model = CrossModalModel(self.config)

        # strip 'module.' si checkpoint DataParallel
        state = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        cross_modal_model.load_state_dict(state, strict=True)

        pretrained_encoder = cross_modal_model.imu_encoder
        encoder_probe = copy.deepcopy(pretrained_encoder)
        encoder_ft    = copy.deepcopy(pretrained_encoder)

        results = {}

        if mode in ["linear_probe", "both"]:
            print("\n--- Linear Probing ---")
            classifier_probe = IMUClassifier(encoder_probe, self.config, freeze_encoder=True)
            print_model_info(classifier_probe, "Classifier (Linear Probe)")

            trainer_probe = ClassificationTrainer(
                classifier_probe, self.config, device=str(self.device), mode="linear_probe"
            )
            trainer_probe.fit(clf_loaders["train"], clf_loaders["val"])

            evaluator_probe = Evaluator(classifier_probe, self.config, device=str(self.device))
            results["linear_probe"] = evaluator_probe.evaluate(clf_loaders["test"])

        if mode in ["finetune", "both"]:
            print("\n--- Full Finetuning ---")
            classifier_ft = IMUClassifier(encoder_ft, self.config, freeze_encoder=False)
            print_model_info(classifier_ft, "Classifier (Finetune)")

            trainer_ft = ClassificationTrainer(
                classifier_ft, self.config, device=str(self.device), mode="finetune"
            )
            trainer_ft.fit(clf_loaders["train"], clf_loaders["val"])

            evaluator_ft = Evaluator(classifier_ft, self.config, device=str(self.device))
            results["finetune"] = evaluator_ft.evaluate(clf_loaders["test"])

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
        import torch
        import pandas as pd

        print("\n" + "=" * 60)
        print("ÉTAPE 4: ÉVALUATION COMPLÈTE")
        print("=" * 60)

        train_meta = self.config.paths.preprocessed_dir / "train_metadata.csv"
        test_meta  = self.config.paths.preprocessed_dir / "test_metadata.csv"

        if not train_meta.exists() or not test_meta.exists():
            raise FileNotFoundError(
                "Metadata introuvable. Lance d'abord le preprocessing."
            )

        train_df = pd.read_csv(train_meta)
        test_df  = pd.read_csv(test_meta)

        checkpoint_path = (
            self.config.paths.checkpoints_dir / "cross_modal" / "best_model.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Checkpoint introuvable. Lance d'abord le pretraining."
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        cross_modal_model = CrossModalModel(self.config)

        state = checkpoint.get("model_state_dict", checkpoint)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        cross_modal_model.load_state_dict(state, strict=False)

        device = torch.device(self.device)
        pretrained_encoder = cross_modal_model.imu_encoder.to(device)
        pretrained_encoder.eval()

        print("\n--- Few-Shot Evaluation ---")
        fs_evaluator = FewShotEvaluator(self.config, device=str(device))

        fs_results = fs_evaluator.run_few_shot_experiments(
            pretrained_encoder=pretrained_encoder,
            train_df=train_df,
            test_df=test_df,
            experiment_name="cross_modal_pretrained",
        )

        fs_agg = fs_evaluator.aggregate_results(fs_results)

        results_dir = self.config.paths.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        fs_results.to_csv(results_dir / "fewshot_results_raw.csv", index=False)
        fs_agg.to_csv(results_dir / "fewshot_results_agg.csv", index=False)

        print("\n✓ Évaluation terminée")
        return fs_results, fs_agg

    def run_all(self):
        print("\n" + "=" * 60)
        print("PIPELINE COMPLET - DÉBUT")
        print("=" * 60)

        if not (self.config.paths.preprocessed_dir / "train_metadata.csv").exists():
            self.run_preprocessing()
        else:
            print("\n✓ Preprocessing déjà effectué")

        if not (self.config.paths.checkpoints_dir / "cross_modal" / "best_model.pt").exists():
            self.run_pretraining()
        else:
            print("\n✓ Pretraining déjà effectué")

        self.run_classification(mode="both")
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
        from src.utils import create_results_summary  # chemin correct

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

# TON CODE COMPLET ICI

