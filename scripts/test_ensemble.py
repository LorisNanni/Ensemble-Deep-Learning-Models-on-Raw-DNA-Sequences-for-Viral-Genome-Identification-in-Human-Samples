import argparse
from src.dataset.csv_dataset import CSVDataset
from src.model.network import Merger, Branch
from src.utils.metrics import auroc as score_fn

import os
import torch
from tqdm import tqdm
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_file")
    parser.add_argument("--model_path", "-m",  required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size for testing (default: 128).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output during testing.")

    return parser.parse_args()


def main():
    # args = parse_args()

    # print("[i] Starting testing with the following parameters:")
    # print(" - datafile:", args.data_file)
    # print(" - model path:", args.model_path)
    # print(" - batch size:", args.batch_size)

    dataset_path = "../dataset"

    base_path = "model_zoo/onehot"

    paths = [
        "branch/frequency",
        "branch/frequency-paper",
        "branch/lp",
        "branch/pattern",
        "branch/pattern-paper",
        "merger/frequency+pattern+lp",
        "merger/lp+frequency",
        "merger/lp+pattern",
        "merger/viraminer",
        "merger/viraminer-paper",
    ]

    logits = []

    for path in tqdm(paths, desc="Processing models"):
        complete_path = os.path.join(base_path, path, "init+norm", "logits.csv")
        csv_data = pd.read_csv(complete_path)
        logits.append(torch.tensor(csv_data["logit"].values))
    
    logits = torch.stack(logits, dim=1)
    logits = logits.sum(dim=1)

    y_pred_prob = torch.sigmoid(logits)

    test_df = CSVDataset(os.path.join(dataset_path, "fullset_test.csv"), transform="onehot_encoding")
    y_true = test_df.dataset[2].values
    y_true = torch.tensor(y_true, dtype=torch.float)

    assert len(y_true) == len(y_pred_prob), f"Length mismatch: y_true has length {len(y_true)}, but y_pred_prob has length {len(y_pred_prob)}."

    auroc, fpr, tpr = score_fn(y_true, y_pred_prob)

    print("[i] auroc:", auroc)


if __name__ == "__main__":
    main()