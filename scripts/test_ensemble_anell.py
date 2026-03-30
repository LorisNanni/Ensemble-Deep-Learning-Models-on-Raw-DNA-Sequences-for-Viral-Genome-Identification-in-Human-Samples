import argparse
from src.dataset.csv_dataset import CSVDataset
from src.model.network import Merger, Branch
from src.utils.metrics import auroc as score_fn
import glob
import os
import torch
from tqdm import tqdm
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_file")
    parser.add_argument("--folder", "-f", default="model_zoo/onehot", help="Path to the model zoo directory (default: 'model_zoo/onehot').")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output during testing.")

    return parser.parse_args()


def main():
    args = parse_args()

    print("[i] Starting testing with the following parameters:")
    print(" - datafile:", args.data_file)
    print(" - folder  :", args.folder)


    paths = glob.glob(os.path.join(args.folder, "branch_pattern-paper_logits.csv"))
    paths.sort()
    print(f"[i] Found {len(paths)}  logit files:")


    logits = []

    for complete_path in tqdm(paths, desc="Processing models"):
        csv_data = pd.read_csv(complete_path)
        logits.append(torch.tensor(csv_data["logit"].values))
    
    logits = torch.stack(logits, dim=1)
    logits = logits.sum(dim=1)

    y_pred_prob = torch.sigmoid(logits)

    test_df = CSVDataset(args.data_file, transform="onehot_encoding")
    y_true = test_df.dataset[2].values
    y_true = torch.tensor(y_true, dtype=torch.float)

    assert len(y_true) == len(y_pred_prob), f"Length mismatch: y_true has length {len(y_true)}, but y_pred_prob has length {len(y_pred_prob)}."

    auroc, fpr, tpr = score_fn(y_true, y_pred_prob)

    print("[i] auroc:", auroc)


if __name__ == "__main__":
    main()