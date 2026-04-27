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
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
import os

RDLogger.DisableLog('rdApp.*')

def generate_features(input_path, output_path):
    print(f"--- Démarrage du Feature Engineering ---")

    if not os.path.exists(input_path):
        print(f"Erreur : Le fichier {input_path} est introuvable")
        return
    
    df = pd.read_parquet(input_path)
    print(f"Données chargées : {len(df)} lignes")

    print("Validation des molécules...")
    df = df.dropna(subset=['smiles']).copy()
    df['smiles'] = df['smiles'].astype(str)
    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if isinstance(x, str) else None)

    initial_count = len(df)
    df = df.dropna(subset=['mol']).reset_index(drop=True)
    print(f"Molécules valides : {len(df)} / {initial_count}")

    print("Génération des Morgan Fingerprints...")
    def get_fp(mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = 2, nBits = 2048)
        return np.array(fp)

    fps = np.stack(df['mol'].apply(get_fp))
    fp_columns = [f"bit_{i}" for i in range(2048)]
    df_fps = pd.DataFrame(fps, columns = fp_columns)

    print("Calcul des descripteurs physico-chimiques...")

    df['MW'] = df['mol'].apply(Descriptors.MolWt)
    df['LogP'] = df['mol'].apply(Descriptors.MolLogP)
    df['H_Donors'] = df['mol'].apply(Descriptors.NumHDonors)
    df['H_Acceptors'] = df['mol'].apply(Descriptors.NumHAcceptors)
    df['TPSA'] = df['mol'].apply(Descriptors.TPSA)
    df['Rotatable_Bonds'] = df['mol'].apply(Descriptors.NumRotatableBonds)

    cols_to_keep = ['DRUG_ID', 'DRUG_NAME', 'TARGET', 'smiles', 'MW', 'LogP', 'H_Donors', 
                    'H_Acceptors', 'TPSA', 'Rotatable_Bonds']
    
    final_df = pd.concat([df[cols_to_keep], df_fps], axis=1)

    final_df.to_parquet(output_path, compression='snappy')
    print(f"--- Terminé ! ---")
    print(f"Fichier sauvegardé : {output_path}")
    print(f"Dimensions finales : {final_df.shape}")

if __name__ == "__main__" :
    INPUT_FILE = "data/processed/drugs_with_smiles.parquet"
    OUTPUT_FILE = 'data/processed/drug_features.parquet'
    generate_features(INPUT_FILE, OUTPUT_FILE)