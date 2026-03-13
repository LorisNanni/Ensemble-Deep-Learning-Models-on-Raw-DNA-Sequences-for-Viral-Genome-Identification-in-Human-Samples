import argparse
from src.dataset.csv_dataset import CSVDataset
from src.model.network import Merger, Branch
from src.utils.compute_logits import compute_logits
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_file")
    parser.add_argument("--model_path", "-m",  required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size for testing (default: 128).")
    parser.add_argument("--save_folder", "-s", default=None, help="Folder to save the logits (default: None).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output during testing.")

    return parser.parse_args()


def main():
    args = parse_args()

    print("[i] Starting testing with the following parameters:")
    print(" - datafile:", args.data_file)
    print(" - model path:", args.model_path)
    print(" - batch size:", args.batch_size)
    print(" - save folder:", args.save_folder)

    assert args.data_file.endswith(".csv"), "Data file must be a CSV file."
    assert os.path.isfile(args.data_file), f"Data file '{args.data_file}' does not exist."
    assert args.batch_size > 0, "Batch size must be a positive integer."

    test_df = CSVDataset(f"{args.data_file}", transform="onehot_encoding")
    test_dl = test_df.get_dataloader(batch_size=128, shuffle=False, drop_last=False)

    print("[i] found ", len(test_df), "samples in the test dataset.")

    if "merger" in args.model_path.lower():
        model = Merger(args.model_path)
    elif "branch" in args.model_path.lower():
        model = Branch(args.model_path)
    else:
        ## ask the user: is it a merger or a branch model?
        type_ = input("Is the model a 'merger' or a 'branch' model? (type 'merger' or 'branch'): ").strip().lower()
        while type_ not in ['merger', 'branch']:
            type_ = input("Invalid input. Please type 'merger' or 'branch': ").strip().lower()
        if type_ == 'merger':
            model = Merger(args.model_path)
        else:
            model = Branch(args.model_path)
    
    print(f"[i] initialized successfully a \033[92m{type(model).__name__}\033[0m model.")
    
    compute_logits(model, test_dl, saving_path=args.save_folder, batch_size=args.batch_size)


if __name__ == "__main__":
    main()