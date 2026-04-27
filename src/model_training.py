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
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import tempfile
import os

def train_drug_prediction_model(data_path, model_output_path, experiment_name = "Drug_IC50_prediction"):
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        print("--- Démarrage de l'entraînement du modèle ---")
        
        # 1. Chargement du Master Dataset
        df = pd.read_parquet(data_path)
        print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

        # 2. Préparation des Features (X) et de la Cible (y)
        # On exclut les colonnes d'identification et les métadonnées
        # On garde tout ce qui commence par 'bit_', 'PC_', et nos descripteurs chimiques
        cols_to_exclude = ['DRUG_ID', 'DRUG_NAME', 'CELL_LINE_NAME', 'SANGER_MODEL_ID', 
                        'ModelID', 'CellLineName', 'LN_IC50', 'AUC', 'RMSE', 'smiles', 'pIC50']
        
        X = df.drop(columns=[c for c in cols_to_exclude if c in df.columns])
        y = df['pIC50']
        
        print(f"Nombre de features utilisées : {X.shape[1]}")
        print(X.columns)
        # 3. Split Train / Test
        # On garde 20% des données pour tester la capacité de généralisation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Configuration de LightGBM
        # On utilise des paramètres robustes pour commencer
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1 # Utilise tous les cœurs de ton processeur
        }

        mlflow.log_params(params)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_features", X.shape[1])

        model = lgb.LGBMRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )

        # 6. Évaluation
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print("\n--- RÉSULTATS ---")
        print(f"R² Score : {r2:.4f}  (Plus proche de 1 est mieux)")
        print(f"RMSE     : {rmse:.4f} (Erreur moyenne en unités pIC50)")
        print(f"MAE      : {mae:.4f}")

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # 7. Sauvegarde du modèle et des features
        joblib.dump(model, model_output_path)
        # Important : on sauvegarde la liste des colonnes pour les futures prédictions
        joblib.dump(X.columns.tolist(), "models/model_features_list.pkl")
        
        print(f"\nModèle sauvegardé sous : {model_output_path}")

        # 8. Visualisation rapide (Prediction vs Reality)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.3, color='darkred') # Bordeaux power
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Vérité Terrain (pIC50)')
        plt.ylabel('Prédictions (pIC50)')
        plt.title('Performance du modèle de régression IC50')
        plt.savefig('plots/performance_plot.png')
        print("Graphique de performance généré : plots/performance_plot.png")

if __name__ == "__main__":
    DATA_PATH = "data/processed/master_dataset.parquet"
    MODEL_PATH = "models/drug_predictor_model.pkl"
    train_drug_prediction_model(DATA_PATH, MODEL_PATH)