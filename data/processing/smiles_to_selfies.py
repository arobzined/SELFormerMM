import pandas as pd
from pandarallel import pandarallel
import selfies as sf

def to_selfies(smiles):
    """
    Converts SMILES to SELFIES representation.
    If an error occurs, returns the original SMILES unchanged.
    """
    try:
        return sf.encoder(smiles)
    except sf.EncoderError:
        print(f"EncoderError for SMILES: {smiles}")
        return smiles

def prepare_data(path, save_to):
    """
    Reads a dataset with SMILES, converts SMILES to SELFIES, and saves the result.
    """
    chembl_df = pd.read_csv(path, sep="\t")
    chembl_df["selfies"] = chembl_df["canonical_smiles"]  # Copy the SMILES column

    pandarallel.initialize()
    chembl_df["selfies"] = chembl_df["selfies"].parallel_apply(to_selfies)
    chembl_df.drop(chembl_df[chembl_df["canonical_smiles"] == chembl_df["selfies"]].index, inplace=True)
    chembl_df.drop(columns=["canonical_smiles"], inplace=True)
    chembl_df.to_csv(save_to, index=False)

input_csv_path = "/home/g3bbmproject/main_folder/KG/kg.pt/our_10k_matched_data_with_embeddings.csv"
output_csv_path = "data_with_selfies.csv"
temp_smiles_path = "temp_smiles.csv"
temp_selfies_path = "temp_selfies.csv"

data = pd.read_csv(input_csv_path)

# Save the SMILES column to a temporary file for conversion
data[['smiles']].rename(columns={"smiles": "canonical_smiles"}).to_csv(temp_smiles_path, index=False, sep="\t")

# Convert SMILES to SELFIES using the prepare_data function
prepare_data(path=temp_smiles_path, save_to=temp_selfies_path)

# Load the resulting SELFIES data
selfies_data = pd.read_csv(temp_selfies_path)

# Add the SELFIES column back to the original data
data['selfies'] = selfies_data['selfies']  # Assumes the converted file has a 'selfies' column

# Save the updated data to a new CSV file
data.to_csv(output_csv_path, index=False)

print(f'Total length of data: {len(data)}')
print(f"Updated dataset with SELFIES saved to: {output_csv_path}")