"""
Project: Drug Sensitivity Prediction Pipeline
Copyright (C) 2026 Alexis Hucteau

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import pandas as pd
import numpy as np
import os

def process_targets(input_path, output_path):
    print(f"--- Démarrage du Target Processing ---")
    
    # 1. Chargement du fichier Excel
    if not os.path.exists(input_path):
        print(f"Erreur : Le fichier {input_path} est introuvable.")
        return
    
    # On charge uniquement les colonnes utiles pour alléger la mémoire
    cols_to_load = ['DRUG_ID', 'CELL_LINE_NAME', 'LN_IC50', 'AUC', 'RMSE']
    df = pd.read_excel(input_path, usecols=cols_to_load)
    print(f"Données chargées : {len(df)} mesures d'IC50.")

    # 2. Nettoyage des données manquantes
    initial_count = len(df)
    df = df.dropna(subset=['LN_IC50', 'DRUG_ID']).reset_index(drop=True)
    print(f"Mesures valides après nettoyage des NaNs : {len(df)} / {initial_count}")

    # 3. Conversion LN_IC50 -> pIC50
    # Formule : 6 - (LN_IC50 / ln(10))
    # Note : On suppose que l'IC50 d'origine est en micromolaire (standard GDSC)
    df['pIC50'] = 6 - (df['LN_IC50'] / np.log(10))

    # 4. Filtrage de qualité (Optionnel mais recommandé)
    # On peut filtrer les courbes avec un RMSE trop élevé (mauvaise qualité de fit)
    threshold_rmse = 0.8 # Valeur à ajuster selon tes observations
    df_clean = df[df['RMSE'] < threshold_rmse].copy()
    print(f"Mesures conservées après filtre qualité (RMSE < {threshold_rmse}) : {len(df_clean)}")

    # 5. Sauvegarde
    df_clean.to_parquet(output_path, compression='snappy', index=False)
    print(f"--- Terminé ! ---")
    print(f"Fichier sauvegardé : {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "data/raw/GDSC2_fitted_dose_response_27Oct23.xlsx"
    OUTPUT_FILE = "data/processed/target_clean.parquet"
    process_targets(INPUT_FILE, OUTPUT_FILE)
