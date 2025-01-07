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
        df[label_column_name] = df[label_column_name].apply(lambda x: 0 if x == 'BENIGN' else 1)
        data_df = df.iloc[:, :-1]

        # データフレームの情報を出力
        print("DataFrame info:")
        data_df.info()

        # データの記述統計を出力
        print("\nDataFrame describe:")
        print(data_df.describe(include='all')) # 数値以外の列も表示

        data_np = data_df.values
        print("data_np before normalization:")
        print(data_np)

        # 問題のある値（NaNや無限大）をチェック
        if np.any(np.isnan(data_np)) or np.any(np.isinf(data_np)):
          print("\nWarning: NaN or Inf found in data_np before scaling. They need to be handled before proceeding")
          # NaNや無限大を処理する例（ここではNaNを0で埋める。状況に応じて適切な方法を選択してください。）
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

          # tensorをnumpy配列に戻す
          data_with_label_np = data_tensor.numpy()

          # 新しいデータフレームを作成
          new_df = pd.DataFrame(data_with_label_np, columns=list(data_df.columns) + [label_column_name])

          # CSVファイルへ出力
          new_df.to_csv(output_csv_path, index=False)
          print(f"正規化されたデータが '{output_csv_path}' に保存されました。")

          return data_tensor

        except Exception as e:
            print(f"Error during tensor creation: {e}")
            return None

    except FileNotFoundError:
        print(f"エラー: ファイル '{csv_path}' が見つかりません。")
        return None
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        filename = exc_tb.tb_frame.f_code.co_filename
        lineno = exc_tb.tb_lineno
        print(f"予期せぬエラーが発生しました: {e}")
        print(f"  ファイル: {filename}")
        print(f"  行番号: {lineno}")
        traceback.print_exc()  # tracebackも表示
        return None

def main():
    # 引数パーサーを作成
    parser = argparse.ArgumentParser(description="指定されたファイルを処理するスクリプト")

    # 必須のファイル名引数を追加
    parser.add_argument("input_filename", help="処理対象の入力ファイル名")
    parser.add_argument("output_filename", help="出力ファイル名")

    # 引数を解析
    args = parser.parse_args()

    # ファイル名を取得
    input_filename = args.input_filename
    output_filename = args.output_filename


    # ここでファイル処理を行う
    try:
        # 正規化を実行
        normalized_tensor = normalize_csv_with_torch_label(input_filename, output_filename)

        if normalized_tensor is not None:
            print("正規化されたTensor (ラベル付き):")
            print(normalized_tensor)
    except FileNotFoundError:
        print(f"エラー: ファイル '{input_filename}' が見つかりません。", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
