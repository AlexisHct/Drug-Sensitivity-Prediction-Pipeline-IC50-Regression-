"""
Project: Drug Sensitivity Prediction Pipeline
Copyright (C) 2026 Alexis Hucteau

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import pandas as pd

def bridge_expression_data(expr_path, meta_path, output_path):
    print("--- Création du pont Biologie -> Chimie ---")
    
    df_expr = pd.read_csv(expr_path) 
    
    if 'ModelID' not in df_expr.columns:
        df_expr = df_expr.reset_index().rename(columns={'index': 'ModelID'})

    df_meta = pd.read_csv(meta_path)[['ModelID', 'SangerModelID', 'CellLineName']]
        
    df_bridged = pd.merge(df_expr, df_meta, on='ModelID', how='inner')
    
    df_bridged = df_bridged.rename(columns={'SangerModelID': 'SANGER_MODEL_ID'})

    df_bridged = df_bridged.rename(columns={'CellLineName': 'CELL_LINE_NAME'})

    print(f"Mapping réussi pour {len(df_bridged)} lignées cellulaires.")
    
    df_bridged.to_parquet(output_path, compression='snappy', index=False)
    print(f"Fichier d'expression prêt : {output_path}")

if __name__ == "__main__":
    EXPR_PATH = "data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    MODEL_PATH = "data/raw/Model.csv"
    OUTPUT_PATH = "data/processed/cell_line_expression.parquet"
    bridge_expression_data(EXPR_PATH, MODEL_PATH, OUTPUT_PATH)