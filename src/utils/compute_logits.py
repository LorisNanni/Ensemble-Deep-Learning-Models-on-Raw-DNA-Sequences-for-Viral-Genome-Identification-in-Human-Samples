import torch
from tqdm import tqdm
import pandas as pd

def compute_logits(model, dataloader, saving_path=None, batch_size=128):
    model.set_best_available_device()

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    
    # For scores computation
    logits_collection = torch.empty(0).to(model.device)
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, _ in tqdm(dataloader, desc="Computing logits", total=len(dataloader)):
        
            # Transfer data to correct device
            X = X.to(model.device)
            
            _, logits = model(X)
            
            logits_collection = torch.cat((logits_collection,logits),dim=0)

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