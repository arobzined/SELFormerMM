import os
import pandas as pd
import sys
sys.path.append("/home/g3-bbm-project/main_folder/embedding_codes/text_embed_code.py")

from text_embed_code import process_csv_to_npy

def filter_multi_csv_by_subfolders(multi_csv_path, main_directory, output_dir="filtered_results",
                                  smiles_col_multi="smiles", smiles_col_reference="smiles"):
    """
    Filters a Multi.csv file against each CSV in subfolders of a main directory.
    
    Args:
        multi_csv_path (str): Path to the Multi.csv file
        main_directory (str): Path to directory containing subfolders with reference CSVs
        output_dir (str): Directory to save filtered results (created inside main_directory)
        smiles_col_multi (str): SMILES column name in Multi.csv
        smiles_col_reference (str): SMILES column name in reference CSVs
    """
    try:
        print(f"Main processing started")
        print(f"Multi.csv location: {multi_csv_path}")
        print(f"Reference directory: {main_directory}")
        
        # Read the Multi.csv file once
        print("\nReading Multi.csv...")
        multi_df = pd.read_csv(multi_csv_path)
        multi_df = multi_df.drop_duplicates(subset=smiles_col_multi)
        print(f"Loaded Multi.csv with {len(multi_df)} rows")
        
        # Get all subfolders in the main directory
        subfolders = [f.path for f in os.scandir(main_directory) if f.is_dir()]
        
        if not subfolders:
            print("No subfolders found in the main directory.")
            return
        
        print(f"\nFound {len(subfolders)} subfolders to process")
        
        for folder in subfolders:
            output_path = os.path.join(main_directory, folder)
            os.makedirs(output_path, exist_ok=True)
            smiles_col_reference = 'smiles'
            folder_name = os.path.basename(folder)
            reference_csv = os.path.join(folder, f"{folder_name}.csv")
            output_file = os.path.join(output_path, f"{folder_name}_filtered.csv")
            if folder_name == 'bace':
                smiles_col_reference = 'mol'
            
            print(f"\nProcessing subfolder: {folder_name}")
            
            # Check if reference CSV exists
            if not os.path.exists(reference_csv):
                print(f"  Warning: Expected reference CSV {reference_csv} not found. Skipping.")
                continue
                
            try:
                # Read reference CSV
                ref_df = pd.read_csv(reference_csv)
                ref_smiles = set(ref_df[smiles_col_reference].dropna().unique())
                print(f"  Found {len(ref_smiles)} unique SMILES in reference CSV")
                
                # Filter Multi.csv
                filtered_df = multi_df[multi_df[smiles_col_multi].isin(ref_smiles)]
                
                # Save results
                filtered_df.to_csv(output_file, index=False)
                print(f"  Saved filtered results to {output_file}")
                print(f"  Kept {len(filtered_df)} rows (from original {len(multi_df)})")
                
            except Exception as e:
                print(f"  Error processing {folder_name}: {str(e)}")
                continue
                
        print("\nProcessing complete!")
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Configuration - modify these paths
    multi_csv_path = "/home/g3-bbm-project/main_folder/KG_embedding/data_multi/M^3_Multi.csv"  # Full path to your Multi.csv
    main_directory = "/home/g3-bbm-project/main_folder/FineTune/finetune_data_multi/finetuning_datasets/regression"  # Directory with subfolders containing reference CSVs
    
    # Optional parameters (modify if needed)
    output_dir = "filtered_results"  # Subdirectory to save results
    # smiles_col_multi = "SMILES"    # Uncomment and change if different
    # smiles_col_reference = "SMILES" # Uncomment and change if different
    
    # Run the processing
    filter_multi_csv_by_subfolders(
        multi_csv_path=multi_csv_path,
        main_directory=main_directory,
        output_dir=output_dir
        # Add optional parameters here if needed
    )