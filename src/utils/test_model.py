import torch
from tqdm import tqdm


def test_model(model, dataloader, score_fn, verbose=None):
    model.set_best_available_device(verbose=verbose)

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    
    # For scores computation
    y_pred_prob = torch.empty(0).to(model.device)
    y_true = torch.empty(0).to(model.device)
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for (X, y) in tqdm(dataloader, desc="Testing", total=len(dataloader), disable=(not verbose)):
        
            # Transfer data to correct device
            X = X.to(model.device)
            y = y.to(model.device)
            
            # Reshape labels vector from a row to a column
            if dataloader.batch_size > 1:
                y = y.reshape(-1,1)
            
            pred, _ = model(X)
            
            y_pred_prob = torch.hstack((y_pred_prob,pred.flatten()))
            y_true = torch.hstack((y_true,y.flatten()))
    
    if model.device != 'cpu':
        y_true = y_true.to('cpu')
        y_pred_prob = y_pred_prob.to('cpu')
        
    model.set_to_cpu()

    return score_fn(y_true, y_pred_prob)