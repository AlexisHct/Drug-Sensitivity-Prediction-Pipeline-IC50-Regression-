"""
Project: Drug Sensitivity Prediction Pipeline
Copyright (C) 2026 Alexis Hucteau

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

def apply_pca_to_expression(input_path, output_path, pca_output, gene_name_output, n_components=50):
    df = pd.read_parquet(input_path)
    
    id_cols = ['SANGER_MODEL_ID', 'CELL_LINE_NAME']
    useless_cols =['Unnamed: 0', 'SequencingID', 'ModelConditionID', 'ModelID',
       'IsDefaultEntryForMC', 'IsDefaultEntryForModel']
    df = df.drop(columns=useless_cols)
    ids = df[id_cols]
    genes = df.drop(columns=id_cols)
    
    genes_names = genes.columns.tolist()

    scaler = StandardScaler()
    genes_scaled = scaler.fit_transform(genes)
    
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(genes_scaled)
    
    pc_columns = [f'PC_{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(pca_results, columns=pc_columns)
    
    final_expr = pd.concat([ids.reset_index(drop=True), df_pca], axis=1)
    joblib.dump(pca, pca_output)
    joblib.dump(genes_names, gene_name_output)

    final_expr.to_parquet(output_path)
    print(f"PCA terminée. Variance expliquée : {pca.explained_variance_ratio_.sum():.2%}")
    return final_expr

if __name__ == "__main__":
    INPUT_PATH = "data/processed/cell_line_expression.parquet"
    OUTPUT_PATH = "data/processed/cell_line_pca.parquet"
    PCA_OUTPUT = "models/pca_model.pkl"
    GENE_NAMES = "models/gene_names_list.pkl"
    apply_pca_to_expression(INPUT_PATH, OUTPUT_PATH, PCA_OUTPUT, GENE_NAMES)