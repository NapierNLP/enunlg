import argparse
import torch

import enunlg.util

argparser = argparse.ArgumentParser()
argparser.add_argument("model_files", type=str, nargs="+")

if __name__ == "__main__":
    args = argparser.parse_args()
    for model_file in args.model_files:
        model = torch.load(model_file)
        print(f"Model: {model_file}")
        total_parameters = enunlg.util.count_parameters(model, log_table=False, print_table=True)
