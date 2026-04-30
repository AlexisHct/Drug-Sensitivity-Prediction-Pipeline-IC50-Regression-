"""
Project: Drug Sensitivity Prediction Pipeline
Copyright (C) 2026 Alexis Hucteau

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import pandas as pd
import os

def merge_datasets(drugs_path, targets_path, expr_path, output_path):
    print(f"--- Démarrage du Data Merging ---")
    
    # 1. Chargement des datasets
    if not all(os.path.exists(p) for p in [drugs_path, targets_path, expr_path]):
        print("Erreur : L'un des fichiers d'entrée est introuvable.")
        return
    
    print("Chargement des features chimiques...")
    df_drugs = pd.read_parquet(drugs_path)
    
    print("Chargement des cibles (pIC50)...")
    df_targets = pd.read_parquet(targets_path)

    print("Chargement des expressions géniques...")
    df_expr = pd.read_parquet(expr_path)

    # 2. La Fusion (Inner Join)
    print(f"Fusion des données sur DRUG_ID...")
    final_df = pd.merge(df_targets, df_drugs, on='DRUG_ID', how='inner')
    final_df = pd.merge(final_df, df_expr, on='CELL_LINE_NAME', how='inner')

    # 3. Vérification de la cohérence
    n_drugs_final = final_df['DRUG_ID'].nunique()
    print(f"Nombre de molécules uniques après fusion : {n_drugs_final}")
    print(f"Nombre total de couples (Drogue, Cellule) : {len(final_df)}")

    # 4. Nettoyage des colonnes redondantes
    if 'smiles_y' in final_df.columns:
        final_df = final_df.drop(columns=['smiles_y']).rename(columns={'smiles_x': 'smiles'})
    if 'DRUG_NAME_y' in final_df.columns:
        final_df = final_df.drop(columns=['DRUG_NAME_y']).rename(columns={'DRUG_NAME_x': 'DRUG_NAME'})

    # 5. Sauvegarde du Master Dataset
    print(f"Sauvegarde du dataset final ({final_df.shape[0]} lignes, {final_df.shape[1]} colonnes)...")
    final_df.to_parquet(output_path, compression='snappy', index=False)
    
    print(f"--- Terminé ! ---")
    print(f"Fichier prêt pour l'entraînement : {output_path}")

if __name__ == "__main__":
    DRUGS_FILE = "data/processed/drug_features.parquet"
    TARGETS_FILE = "data/processed/target_clean.parquet"
    EXPR_PATH = "data/processed/cell_line_pca.parquet"
    OUTPUT_FILE = "data/processed/master_dataset.parquet"
    
    merge_datasets(DRUGS_FILE, TARGETS_FILE, EXPR_PATH, OUTPUT_FILE)