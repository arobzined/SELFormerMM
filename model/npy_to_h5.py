import numpy as np
import h5py
import argparse
import os

"""
Example usage:
python convert_npy_to_h5.py /path/to/input_file.npy /path/to/output_file.h5
"""

def convert_npy_to_h5(npy_file_path, h5_output_path):
    if not os.path.isfile(npy_file_path):
        print(f"Error: Input file '{npy_file_path}' does not exist.")
        return

    data = np.load(npy_file_path)

    with h5py.File(h5_output_path, 'w') as h5_file:
        h5_file.create_dataset('data', data=data)
        print(f"Data from '{npy_file_path}' has been successfully saved to '{h5_output_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a .npy file to a .h5 file.")
    parser.add_argument("npy_file", type=str, help="Path to the input .npy file.")
    parser.add_argument("h5_file", type=str, help="Path to the output .h5 file.")

    args = parser.parse_args()
    convert_npy_to_h5(args.npy_file, args.h5_file)
