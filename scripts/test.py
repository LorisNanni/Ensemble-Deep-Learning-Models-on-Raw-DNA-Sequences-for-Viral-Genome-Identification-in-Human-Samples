import argparse
from src.dataset.csv_dataset import CSVDataset
from src.model.network import Merger, Branch
from src.utils.metrics import auroc as score_fn

import os
import torch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_file")
    parser.add_argument("--model_path", "-m",  required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size for testing (default: 128).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output during testing.")

    return parser.parse_args()


def main():
    args = parse_args()

    print("[i] Starting testing with the following parameters:")
    print(" - datafile:", args.data_file)
    print(" - model path:", args.model_path)
    print(" - batch size:", args.batch_size)

    assert args.data_file.endswith(".csv"), "Data file must be a CSV file."
    assert os.path.isfile(args.data_file), f"Data file '{args.data_file}' does not exist."
    assert args.batch_size > 0, "Batch size must be a positive integer."

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
    test_df = CSVDataset(f"{args.data_file}", transform="onehot_encoding")
    test_dl = test_df.get_dataloader(batch_size=args.batch_size, shuffle=False, drop_last=False)

    print("[i] found ", len(test_df), "samples in the test dataset.")
    
    # moving the model to the best available device (GPU if available, otherwise CPU) 
    model.set_best_available_device(verbose=args.verbose)

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    
    # For scores computation
    y_pred_prob = torch.empty(0).to(model.device)
    y_true = torch.empty(0).to(model.device)
    
    with torch.no_grad():
        for (X, y) in tqdm(test_dl, desc="Testing", total=len(test_dl)):
        
            # Transfer data to correct device
            X, y = X.to(model.device), y.to(model.device)
            
            # Reshape labels vector from a row to a column
            if test_dl.batch_size > 1:
                y = y.reshape(-1,1)
            
            # Forward pass through the model to get predicted probabilities
            pred, _ = model(X)
            
            y_pred_prob = torch.hstack((y_pred_prob,pred.flatten()))
            y_true = torch.hstack((y_true,y.flatten()))
    
    if model.device != 'cpu':
        y_true = y_true.to('cpu')
        y_pred_prob = y_pred_prob.to('cpu')
        
    model.set_to_cpu()

    auroc, fpr, tpr = score_fn(y_true, y_pred_prob)

    print("[i] auroc:", auroc)


if __name__ == "__main__":
    main()