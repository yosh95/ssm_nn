import argparse
import sys
import torch
import traceback
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def normalize_csv_with_torch_label(csv_path, output_csv_path):
    try:
        df = pd.read_csv(csv_path)
        label_column_name = df.columns[-1]
        df[label_column_name] = df[label_column_name].apply(
            lambda x: 0 if x == 'BENIGN' else 1)
        data_df = df.iloc[:, :-1]

        print("DataFrame info:")
        data_df.info()

        print("\nDataFrame describe:")
        print(data_df.describe(include='all'))

        data_np = data_df.values
        print("data_np before normalization:")
        print(data_np)

        if np.any(np.isnan(data_np)) or np.any(np.isinf(data_np)):
            print("\nWarning: NaN or Inf found in data_np before scaling. " +
                  "They need to be handled before proceeding")
            data_np = np.nan_to_num(data_np, nan=0.0, posinf=0.0, neginf=0.0)
            print("data_np after removing NaN and inf")
            print(data_np)

        scaler = MinMaxScaler()
        try:
            data_normalized = scaler.fit_transform(data_np)
            print("data_normalized after normalization:")
            print(data_normalized)
        except Exception as e:
            print(f"Error during normalization: {e}")
            return None

        label_np = df[label_column_name].values.reshape(-1, 1)
        print("label_np:")
        print(label_np)

        data_with_label = np.concatenate((data_normalized, label_np), axis=1)
        print("data_with_label before tensor conversion:")
        print(data_with_label)

        try:
            data_tensor = torch.tensor(data_with_label, dtype=torch.float32)

            data_with_label_np = data_tensor.numpy()

            new_df = pd.DataFrame(data_with_label_np,
                                  columns=list(data_df.columns) +
                                  [label_column_name])

            new_df.to_csv(output_csv_path, index=False)
            print(f"Normalized data has been saved to '{output_csv_path}'.")

            return data_tensor

        except Exception as e:
            print(f"Error during tensor creation: {e}")
            return None

    except FileNotFoundError:
        print(f"Error: '{csv_path}' is not found.")
        return None
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        filename = exc_tb.tb_frame.f_code.co_filename
        lineno = exc_tb.tb_lineno
        print(f"Unexpected error: {e}")
        print(f"  File: {filename}")
        print(f"  Line No.: {lineno}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_filename")
    parser.add_argument("output_filename")

    args = parser.parse_args()

    input_filename = args.input_filename
    output_filename = args.output_filename

    try:
        normalized_tensor = normalize_csv_with_torch_label(input_filename,
                                                           output_filename)

        if normalized_tensor is not None:
            print("Normalized Tensor:")
            print(normalized_tensor)
    except FileNotFoundError:
        print(f"Erro: '{input_filename}' is not found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
