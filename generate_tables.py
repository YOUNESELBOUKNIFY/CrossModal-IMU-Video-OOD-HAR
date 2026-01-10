"""
Génération de tableaux comparatifs style article
Réplique exactement les Tables 3 et 4 de l'article avec format "mean ± std"
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json


class TableGenerator:
    """
    Générateur de tableaux pour les résultats
    Format similaire aux Tables 3 et 4 de l'article
    """
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def format_mean_std(self, mean: float, std: float, decimals: int = 2) -> str:
        """
        Formate les résultats en "mean ± std"
        """
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
    
    def generate_table3_style(self, results_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Génère un tableau style Table 3 de l'article
        Few-shot performance comparison
        
        Expected columns in results_df:
        - experiment: nom de l'expérience (ex: "cross_modal_pretrained")
        - n_samples: nombre de samples par classe (10, 20, 50, 100)
        - mode: 'linear_probe' ou 'finetune'
        - run: numéro du run (0-4)
        - balanced_accuracy: score (0-100)
        - f1_macro: score (0-100)
        - accuracy: score (0-100)
        
        Returns:
            dict avec plusieurs tableaux formatés
        """
        
        # Grouper et calculer mean/std
        grouped = results_df.groupby(['experiment', 'n_samples', 'mode']).agg({
            'balanced_accuracy': ['mean', 'std'],
            'f1_macro': ['mean', 'std'],
            'accuracy': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['_'.join(str(col)).strip('_') for col in grouped.columns.values]
        
        # Renommer pour clarté
        grouped.columns = [col.replace('balanced_accuracy', 'bal_acc') for col in grouped.columns]
        
        # Format mean ± std pour chaque métrique
        for metric in ['bal_acc', 'f1_macro', 'accuracy']:
            grouped[f'{metric}_formatted'] = grouped.apply(
                lambda row: self.format_mean_std(
                    row[f'{metric}_mean'],
                    row[f'{metric}_std']
                ), axis=1
            )
        
        # Créer pivot tables pour chaque métrique
        pivots = {}
        
        for metric, formatted_col in [
            ('balanced_accuracy', 'bal_acc_formatted'),
            ('f1_macro', 'f1_macro_formatted'),
            ('accuracy', 'accuracy_formatted')
        ]:
            pivot = grouped.pivot_table(
                index=['experiment', 'mode'],
                columns='n_samples',
                values=formatted_col,
                aggfunc='first'
            )
            pivot.columns.name = '# labels'
            pivots[metric] = pivot
        
        # Créer aussi un tableau détaillé avec toutes les métriques
        pivots['detailed'] = grouped
        
        return pivots
    
    def generate_table4_style(self, zero_shot_results: Dict[str, Dict[str, tuple]]) -> pd.DataFrame:
        """
        Génère un tableau pour les résultats zero-shot
        Style Table 4 de l'article
        
        Args:
            zero_shot_results: dict avec structure:
            {
                'Ego4D → PD': {
                    'B. Acc.': (mean, std),
                    'F1': (mean, std),
                    'MRR': (mean, std),
                    'R@1': (mean, std),
                    'R@3': (mean, std)
                },
                ...
            }
        
        Returns:
            DataFrame formaté style article
        """
        rows = []
        
        for experiment, metrics in zero_shot_results.items():
            row = {'Experiment': experiment}
            
            for metric_name, (mean, std) in metrics.items():
                row[metric_name] = self.format_mean_std(mean, std, decimals=3)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.set_index('Experiment')
        
        return df
    
    def create_comparison_table(self, 
                                experiments_results: Dict[str, pd.DataFrame],
                                metric: str = 'balanced_accuracy') -> pd.DataFrame:
        """
        Compare plusieurs expériences côte à côte
        
        Args:
            experiments_results: dict avec {experiment_name: results_df}
            metric: métrique à comparer
        
        Returns:
            DataFrame comparatif
        """
        all_tables = []
        
        for exp_name, results_df in experiments_results.items():
            tables = self.generate_table3_style(results_df)
            table = tables[metric].copy()
            table['Experiment'] = exp_name
            all_tables.append(table)
        
        # Concat et réorganiser
        comparison = pd.concat(all_tables)
        comparison = comparison.reset_index()
        
        # Pivot pour avoir un format plus lisible
        final = comparison.pivot_table(
            index=['Experiment', 'mode'],
            columns='n_samples',
            values=comparison.columns[-1],  # La colonne avec les valeurs
            aggfunc='first'
        )
        
        return final
    
    def save_tables(self, tables: Dict[str, pd.DataFrame], prefix: str = 'table'):
        """
        Sauvegarde les tableaux dans plusieurs formats
        """
        saved_files = []
        
        for name, table in tables.items():
            base_filename = f'{prefix}_{name}'
            
            # 1. CSV (facile à réutiliser)
            csv_path = self.results_dir / f'{base_filename}.csv'
            table.to_csv(csv_path)
            saved_files.append(str(csv_path))
            
            # 2. LaTeX (pour paper)
            latex_path = self.results_dir / f'{base_filename}.tex'
            latex_str = table.to_latex(
                escape=False,
                caption=f"Table: {name}",
                label=f"tab:{prefix}_{name}",
                column_format='l' + 'c' * len(table.columns)
            )
            with open(latex_path, 'w') as f:
                f.write(latex_str)
            saved_files.append(str(latex_path))
            
            # 3. Markdown (pour README)
            md_path = self.results_dir / f'{base_filename}.md'
            with open(md_path, 'w') as f:
                f.write(table.to_markdown())
            saved_files.append(str(md_path))
        
        print(f"\n✓ {len(saved_files)} fichiers sauvegardés dans {self.results_dir}")
        for f in saved_files[:5]:  # Afficher les 5 premiers
            print(f"  - {Path(f).name}")
        if len(saved_files) > 5:
            print(f"  ... et {len(saved_files) - 5} autres")
        
        return saved_files


def create_article_tables_from_results(config):
    """
    Fonction principale pour créer tous les tableaux depuis les résultats sauvegardés
    """
    results_dir = config.paths.results_dir
    generator = TableGenerator(results_dir)
    
    print("\n" + "="*60)
    print("GÉNÉRATION DES TABLEAUX STYLE ARTICLE")
    print("="*60)
    
    all_tables = {}
    
    # ========== TABLE 3: Few-shot Performance ==========
    print("\n1. Table 3: Few-shot Performance...")
    
    try:
        # Charger résultats few-shot
        fs_results_path = results_dir / 'fewshot_results_raw.csv'
        
        if fs_results_path.exists():
            fs_results = pd.read_csv(fs_results_path)
            
            # Générer tableaux
            fs_tables = generator.generate_table3_style(fs_results)
            
            print("\n--- Few-shot Balanced Accuracy ---")
            print(fs_tables['balanced_accuracy'].to_string())
            
            print("\n--- Few-shot F1 Macro ---")
            print(fs_tables['f1_macro'].to_string())
            
            # Sauvegarder
            generator.save_tables({
                'fewshot_balanced_acc': fs_tables['balanced_accuracy'],
                'fewshot_f1_macro': fs_tables['f1_macro'],
                'fewshot_accuracy': fs_tables['accuracy'],
                'fewshot_detailed': fs_tables['detailed']
            }, prefix='table3')
            
            all_tables['table3'] = fs_tables
        else:
            print(f"⚠️  Fichier non trouvé: {fs_results_path}")
    
    except Exception as e:
        print(f"⚠️  Erreur lors de la génération Table 3: {e}")
    
    # ========== TABLE 4: Zero-shot Performance ==========
    print("\n2. Table 4: Zero-shot Performance...")
    
    try:
        # Charger résultats zero-shot (si sauvegardés)
        zero_shot_path = results_dir / 'zeroshot_results.json'
        
        if zero_shot_path.exists():
            with open(zero_shot_path, 'r') as f:
                zero_shot_data = json.load(f)
            
            # Générer tableau
            zero_shot_table = generator.generate_table4_style(zero_shot_data)
            
            print("\n--- Zero-shot Performance ---")
            print(zero_shot_table.to_string())
            
            # Sauvegarder
            generator.save_tables({
                'zeroshot': zero_shot_table
            }, prefix='table4')
            
            all_tables['table4'] = zero_shot_table
        else:
            print(f"⚠️  Fichier non trouvé: {zero_shot_path}")
            print("    (Zero-shot results doivent être sauvegardés manuellement)")
    
    except Exception as e:
        print(f"⚠️  Erreur lors de la génération Table 4: {e}")
    
    # ========== TABLE 5: Ablation Study ==========
    print("\n3. Table 5: Ablation Study...")
    
    try:
        ablation_path = results_dir / 'ablation_results.csv'
        
        if ablation_path.exists():
            ablation_df = pd.read_csv(ablation_path, index_col=0)
            
            print("\n--- Ablation Study ---")
            print(ablation_df.to_string())
            
            generator.save_tables({
                'ablation': ablation_df
            }, prefix='table5')
            
            all_tables['table5'] = ablation_df
        else:
            print(f"⚠️  Fichier non trouvé: {ablation_path}")
    
    except Exception as e:
        print(f"⚠️  Erreur lors de la génération Table 5: {e}")
    
    # ========== Comparaison globale ==========
    print("\n4. Tableau de comparaison global...")
    
    try:
        comparison_path = results_dir / 'classification_comparison.csv'
        
        if comparison_path.exists():
            comparison_df = pd.read_csv(comparison_path, index_col=0)
            
            print("\n--- Comparaison Linear Probe vs Finetune ---")
            print(comparison_df.to_string())
            
            generator.save_tables({
                'comparison': comparison_df
            }, prefix='comparison')
            
            all_tables['comparison'] = comparison_df
        else:
            print(f"⚠️  Fichier non trouvé: {comparison_path}")
    
    except Exception as e:
        print(f"⚠️  Erreur lors de la comparaison: {e}")
    
    # ========== Résumé final ==========
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    print(f"Tableaux générés: {len(all_tables)}")
    for table_name in all_tables.keys():
        print(f"  ✓ {table_name}")
    
    print(f"\nFichiers sauvegardés dans: {results_dir}")
    
    return all_tables


def create_latex_paper_table(df: pd.DataFrame, 
                             caption: str, 
                             label: str,
                             save_path: Path,
                             use_booktabs: bool = True):
    """
    Crée un tableau LaTeX publication-ready avec booktabs
    """
    # Début du tableau
    latex_str = "\\begin{table}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += f"\\label{{{label}}}\n"
    
    # Column format
    ncols = len(df.columns)
    col_format = "l" + "c" * ncols
    
    if use_booktabs:
        latex_str += "\\begin{tabular}{" + col_format + "}\n"
        latex_str += "\\toprule\n"
    else:
        latex_str += "\\begin{tabular}{|" + "|".join(["l"] + ["c"]*ncols) + "|}\n"
        latex_str += "\\hline\n"
    
    # Header
    header = " & ".join([""] + [str(c) for c in df.columns]) + " \\\\"
    latex_str += header + "\n"
    
    if use_booktabs:
        latex_str += "\\midrule\n"
    else:
        latex_str += "\\hline\n"
    
    # Rows
    for idx, row in df.iterrows():
        row_str = str(idx)
        for val in row:
            if isinstance(val, (int, float)):
                row_str += f" & {val:.2f}"
            else:
                row_str += f" & {val}"
        latex_str += row_str + " \\\\\n"
    
    # End
    if use_booktabs:
        latex_str += "\\bottomrule\n"
    else:
        latex_str += "\\hline\n"
    
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table}\n"
    
    # Sauvegarder
    with open(save_path, 'w') as f:
        f.write(latex_str)
    
    print(f"✓ LaTeX table sauvegardée: {save_path}")
    
    return latex_str


def format_results_for_readme(results_dir: Path) -> str:
    """
    Crée une section formatée pour le README avec tous les résultats
    """
    readme_section = "## 📊 Résultats\n\n"
    
    # Table 3: Few-shot
    fs_path = results_dir / 'table3_fewshot_balanced_acc.md'
    if fs_path.exists():
        readme_section += "### Few-Shot Performance (Balanced Accuracy)\n\n"
        with open(fs_path, 'r') as f:
            readme_section += f.read() + "\n\n"
    
    # Comparaison
    comp_path = results_dir / 'comparison_comparison.md'
    if comp_path.exists():
        readme_section += "### Linear Probing vs Full Finetuning\n\n"
        with open(comp_path, 'r') as f:
            readme_section += f.read() + "\n\n"
    
    return readme_section


def main():
    """Test de la génération de tableaux avec données fictives"""
    import sys
    sys.path.append('.')
    
    try:
        from config import CONFIG
        results_dir = CONFIG.paths.results_dir
    except:
        results_dir = Path('./outputs/results')
        results_dir.mkdir(parents=True, exist_ok=True)
    
    print("Test de génération de tableaux avec données fictives...")
    
    # Créer des données fictives
    np.random.seed(42)
    
    test_data = []
    for experiment in ['IMU-only SSL', 'IMU2CLIP', 'Ours']:
        for n_samples in [10, 20, 50, 100]:
            for mode in ['linear_probe', 'finetune']:
                for run in range(5):
                    # Simuler des performances réalistes
                    if experiment == 'Ours':
                        base_acc = 85 + n_samples * 0.1
                    elif experiment == 'IMU-only SSL':
                        base_acc = 60 + n_samples * 0.15
                    else:  # IMU2CLIP
                        base_acc = 35 + n_samples * 0.05
                    
                    test_data.append({
                        'experiment': experiment,
                        'n_samples': n_samples,
                        'mode': mode,
                        'run': run,
                        'balanced_accuracy': base_acc + np.random.randn() * 2,
                        'f1_macro': base_acc - 2 + np.random.randn() * 1.5,
                        'accuracy': base_acc + 1 + np.random.randn() * 2.5
                    })
    
    test_df = pd.DataFrame(test_data)
    
    # Générer tableaux
    generator = TableGenerator(results_dir)
    tables = generator.generate_table3_style(test_df)
    
    print("\n--- Balanced Accuracy ---")
    print(tables['balanced_accuracy'])
    
    print("\n--- F1 Macro ---")
    print(tables['f1_macro'])
    
    # Sauvegarder
    generator.save_tables({
        'test_balanced_acc': tables['balanced_accuracy'],
        'test_f1_macro': tables['f1_macro']
    }, prefix='test')
    
    # Test LaTeX
    latex_path = results_dir / 'test_latex_table.tex'
    create_latex_paper_table(
        tables['balanced_accuracy'].iloc[:4],  # Premières lignes
        caption="Few-shot Performance (Test Data)",
        label="tab:test",
        save_path=latex_path
    )
    
    print("\n✓ Test terminé avec succès!")


if __name__ == "__main__":
    main()