import argparse
from src.dataset.csv_dataset import CSVDataset
from src.model.network import Merger, Branch
from src.utils.compute_logits import compute_logits

import os
import torch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_file")
    parser.add_argument("--model_zoo_path", "-m",  default="model_zoo/onehot", help="Path to the model zoo directory (default: 'model_zoo/onehot').")
    parser.add_argument("--initialization", "-i", default="init+norm", help="Model initialization to use (default: 'init+norm').")
    parser.add_argument("--save_folder", "-s", default=None, help="Folder to save the logits (default: None).")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size for testing (default: 128).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output during testing.")

    return parser.parse_args()


def main(noise, seed):
    args = parse_args()

    print("[i] Starting testing with the following parameters:")
    print(" - datafile:", args.data_file)
    print(" - model zoo path:", args.model_zoo_path)
    print(" - initialization:", args.initialization)
    print(" - batch size:", args.batch_size)
    if args.save_folder is None:
        print(" - save folder: None (logits will be saved in the same folder as the model at logits.csv)")
    else:
        print(" - save folder:", args.save_folder)

    # assert args.data_file.endswith(".csv"), "Data file must be a CSV file."
    # assert os.path.isfile(args.data_file), f"Data file '{args.data_file}' does not exist."
    assert args.batch_size > 0, "Batch size must be a positive integer."
    path = os.path.join("/home/velazquez/viralminer/dataset/noisy_datasets", f"noise_{noise}", f"seed_{seed}", f"{args.data_file}")
    test_df = CSVDataset(f"{path}", transform="onehot_encoding")
    test_dl = test_df.get_dataloader(batch_size=args.batch_size, shuffle=False, drop_last=False)
    print("[i] found ", len(test_df), "samples in the test dataset", path)

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
    
    for path_idx, path in enumerate(paths):

        modelptm = os.path.join(args.model_zoo_path, path, "init+norm", "model.ptm")
        # if args.save_folder is None:
        #     savepath = os.path.join(args.model_zoo_path, path, "init+norm", "logits.csv")
        # else:
        #savepath = os.path.join(args.save_folder, f"{path.replace('/', '_')}_logits_{}.csv")
        savepath = os.path.join("/home/velazquez/viralminer/dataset/noisy_datasets/out", f"noise_{noise}", f"seed_{seed}", path, "logits.csv")
        os.makedirs(os.path.dirname(savepath), exist_ok=True)

        print(f"[i] {path_idx+1}/{len(paths)} processing model: {modelptm}")
        assert os.path.isfile(modelptm), f"Model file '{modelptm}' does not exist."

        if "merger" in modelptm.lower():
            model = Merger(modelptm)
        elif "branch" in modelptm.lower():
            model = Branch(modelptm)
        else:
            raise ValueError(f"Model path '{modelptm}' does not contain 'merger' or 'branch'. Please specify a valid model path.")

        print(f"[i] successfully initialized a \033[92m{type(model).__name__}\033[0m model.")
        compute_logits(model, test_dl, saving_path=savepath, batch_size=args.batch_size)
    print()
    print("[i] All models processed successfully.")

if __name__ == "__main__":
    for noise in [1, 5, 10]:
        for seed in [1, 2, 3, 4, 5]:
            seed_local = noise*1000 + seed
            print(f"\n=== Processing noise={noise}%, seed={seed_local} ===")
            main(noise, seed_local)