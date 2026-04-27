"""
Project: Drug Sensitivity Prediction Pipeline
Copyright (C) 2026 Alexis Hucteau

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import pandas as pd
import pubchempy as pcp
from tqdm import tqdm
import os

def fetch_smiles_from_pubchem(drug_name):

    if pd.isna(drug_name) or drug_name == "":
        return None
    
    try:
        compounds = pcp.get_compounds(drug_name, 'name')

        if compounds:
            return compounds[0].connectivity_smiles
        else:
            return None
        
    except Exception as e:
        return None
    
def process_drug_metadata(df_compounds):

    tqdm.pandas()

    df_compounds['smiles'] = df_compounds['DRUG_NAME'].progress_apply(fetch_smiles_from_pubchem)

    found = df_compounds['smiles'].notna().sum()
    total = len(df_compounds)
    print(f"Terminé ! {found}/{total} SMILES récupérés")

    return df_compounds

def ingestion_phase_1(ic50_path, compound_path, output_folder="data/processed"):

    print(f"Lecture des données {ic50_path} et {compound_path}...")
    df_ic50 = pd.read_excel(ic50_path, engine="openpyxl")
    df_compounds = pd.read_csv(compound_path)

    needed_drugs = df_ic50['DRUG_ID'].unique()

    drugs_metadata = df_compounds[df_compounds['DRUG_ID'].isin(needed_drugs)].copy()

    print("Récupération des SMILES")

    drugs_metadata = process_drug_metadata(drugs_metadata)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Dossier créé : {output_folder}")

    output_path = os.path.join(output_folder, "drugs_with_smiles.parquet")

    print(f"Sauvegarde des données vers {output_path}...")

    drugs_metadata.to_parquet(output_path, engine='pyarrow', index=False)

    print(f"Sauvergarde Terminée !")
    
    return output_path

if __name__ == "__main__":
    ingestion_phase_1(
        ic50_path="data/raw/GDSC2_fitted_dose_response_27Oct23.xlsx",
        compound_path="data/raw/screened_compounds_rel_8.5.csv"
    )

