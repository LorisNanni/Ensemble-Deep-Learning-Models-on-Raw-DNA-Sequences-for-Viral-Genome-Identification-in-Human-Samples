import torch
from tqdm import tqdm
import pandas as pd
import time

def compute_logits(model, dataloader, saving_path=None, batch_size=128):
    model.set_best_available_device()

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    
    # For scores computation
    logits_collection = torch.empty(0).to(model.device)
    times = []
    tot_= 0
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, _ in tqdm(dataloader, desc="Computing logits", total=len(dataloader)):
        
            # Transfer data to correct device
            X = X.to(model.device)
            
            torch.cuda.synchronize()  # Wait for all previous CUDA operations to finish
            s = time.time()
            _, logits = model(X)
            torch.cuda.synchronize()  # Wait for the events to be recorded
            e = time.time()
            times.append(e - s)
            tot_ += X.size(0)

            logits_collection = torch.cat((logits_collection,logits),dim=0)
    
    avg_ = sum(times)/tot_
    print(f"[i] Average time per sample: {avg_:.6f} seconds")
    
    # simulate 100K samplesù
    avg_100k = avg_ * 100000
    print(f"[i] Estimated time for 100K samples: {avg_100k:.2f} seconds ({avg_100k/3600:.2f} hours)")

    with open("timing_log.txt", "a") as f:
        f.write(f"saving path: {saving_path}\n")
        f.write(f"Computed logits for {tot_} samples in {sum(times):.2f} seconds (average: {avg_:.6f} seconds/sample)\n")
        f.write(f"Estimated time for 100K samples: {avg_100k:.2f} seconds ({avg_100k/3600:.2f} hours)\n")
        f.write("-" * 50 + "\n")

    if(model.device != 'cpu'): 
        logits_collection = logits_collection.to('cpu')
        model.set_to_cpu()

    # Get the labels
    # labels = torch.tensor(dataloader.dataset.iloc[:, -1].values)

    # If no saving_path, return
    if saving_path != None:
        # Retrieve the full dataset
        # dataloader.dataset is a Pandas DataFrame, we can retrieve the full dataset with dataloader.dataset.dataset
        data = dataloader.dataset.dataset
        
        # Get the corresponding logits
        logits = pd.Series(logits_collection.flatten())
        # Concat the dataset with the logits
        new_data = pd.concat([data,logits], axis=1)
        # Save the dataframe into the file
        new_data.columns = ['sample', 'y_true', 'logit']
        print("[i] saving the logits to:", saving_path)
        new_data.to_csv(saving_path, index=False)
        print("[i] logits saved successfully.")
    else:
        print("[i] no saving path provided. Ending peacefully")

    return logits_collection, saving_path