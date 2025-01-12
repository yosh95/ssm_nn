#!/usr/bin/env python3

import argparse
import sys
import torch
import traceback
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler


def normalize_csv_with_torch_label(csv_path,
                                   output_csv_path,
                                   label_mapping_json):
    try:
        df = pd.read_csv(csv_path)
        label_column_name = df.columns[-1]

        with open(label_mapping_json, 'r') as f:
            label_mapping = json.load(f)

        df[label_column_name] = df[label_column_name].apply(
            lambda x: label_mapping.get(x, -1))

        data_df = df.iloc[:, :-1]

        data_tensor = torch.tensor(data_df.values, dtype=torch.float32)

        if torch.isnan(data_tensor).any() or torch.isinf(data_tensor).any():
            print("Warning: NaN or Inf found in data_tensor before " +
                  "scaling. Handling...")
            data_tensor = torch.nan_to_num(data_tensor,
                                           nan=0.0,
                                           posinf=0.0,
                                           neginf=0.0)

        scaler = MinMaxScaler()
        try:
            data_scaled = scaler.fit_transform(data_tensor.numpy())
            data_normalized = torch.tensor(data_scaled, dtype=torch.float32)
        except Exception as e:
            print(f"Error during normalization: {e}")
            return None

        label_tensor = torch.tensor(
            df[label_column_name].values.reshape(-1, 1),
            dtype=torch.float32)

        data_with_label = torch.cat((data_normalized, label_tensor), dim=1)

        try:
            data_with_label_np = data_with_label.numpy()
            new_df = pd.DataFrame(data_with_label_np,
                                  columns=list(data_df.columns) +
                                  [label_column_name])
            new_df.to_csv(output_csv_path, index=False)
            print(f"Normalized data has been saved to '{output_csv_path}'.")

            return data_with_label

        except Exception as e:
            print(f"Error during tensor creation: {e}")
            return None

    except FileNotFoundError as e:
        print(f"Error: {e}")
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
    parser.add_argument("label_mapping_json")

    args = parser.parse_args()

    input_filename = args.input_filename
    output_filename = args.output_filename
    label_mapping_json = args.label_mapping_json

    try:
        normalize_csv_with_torch_label(input_filename,
                                       output_filename,
                                       label_mapping_json)
    except FileNotFoundError:
        print(f"Erro: '{input_filename}' is not found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
