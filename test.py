import pandas as pd

df = pd.read_parquet("data/processed/drugs_with_smiles.parquet")

print(f"Nombre de lignes : {len(df)}")
print(f"Nombre de SMILES trouvés : {df['smiles'].notna().sum()}")
print(df[['DRUG_NAME', 'smiles']].head())