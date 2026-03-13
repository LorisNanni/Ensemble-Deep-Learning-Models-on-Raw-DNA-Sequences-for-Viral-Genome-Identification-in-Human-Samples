## for train
import sys
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from src.utils.test_model import test_model


def train_model(model, train_dl, loss_fn, optimizer, scheduler, epochs,
                epsilon=0, min_score=0,
                validation_fn=None, val_dl=None, verbose=0):
    
    if validation_fn:
        score_collection = []
        best_model = None
        best_score = -1
        best_epoch = 0
    
    # Early stopping counters
    no_improvement = 0 # if no improvement of epsilon after 6 epochs
    no_over_min_score = 0 # if no score > min_score after 6 epochs

    for epoch in range(epochs):       
        show_device = True if epoch == 0 else 0
        model.set_best_available_device(verbose=show_device)
        # Train
        # loss = model.__train_one_epoch(train_dl, loss_fn, optimizer, scheduler, epoch)
        with torch.autograd.set_detect_anomaly(True): loss = train_one_epoch(model, train_dl, loss_fn, optimizer, scheduler, epoch)

        # Validate
        if validation_fn:
            score, _, _ = test_model(model, val_dl, validation_fn)
            
            output_string = f"Epoch {str(epoch+1).zfill(2)} - Loss: {loss:>2.5e}"
            output_string += f" - Validation Score: {score:>.5f}"
            print(output_string)
        
            # Save score for plot
            score_collection.append(score)
            
            # Check score improvement > epsilon
            if (score - best_score) > epsilon:
                best_model = copy.deepcopy(model.state_dict())
                best_score = score
                best_epoch = epoch+1
                no_improvement = 0
            else:
                no_improvement += 1
            # Check if score > min_score
            if (score <= min_score):
                no_over_min_score += 1
            else:
                no_over_min_score = 0
            
            # Check if to early stop
            if no_improvement > 6:
                print(f"Training stopped at epoch: #{str(epoch+1).zfill(2)} beacuse no Score improvement after 6 epochs (with error: {epsilon:>.5f})")
                break    
            # If no improvemnt than min_score after 6 epochs   
            if no_over_min_score > 6:
                print(f"Training stopped at epoch: #{str(epoch+1).zfill(2)} beacuse no Score >= {min_score:>.5f} after 6 epochs")
                break

    if validation_fn:
        model.load_state_dict(best_model)
        model.set_to_cpu()
        
        print(f"Best model at epoch: #{str(best_epoch).zfill(2)} with Validation Score = {best_score:>.5f}")
        print()
    
        return score_collection
    
    model.set_to_cpu()
    
def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, epoch):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    num_batches = len(dataloader)

    # If True, the step is taken at the end of each batch; otherwise at the end of the epoch
    scheduler_step_flag = isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

    # To detect gradient anomalies
    torch.autograd.set_detect_anomaly(True)

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader), desc=f"Epoch {str(epoch+1).zfill(2)}", total=len(dataloader)):
        
        # Set the gradients to zero
        optimizer.zero_grad()
        # Transfer data to correct device
        X = X.to(model.device)
        y = y.to(model.device)
        
        # Reshape labels vector from a row to a column
        if batch_size > 1:
            y = y.reshape(-1,1)

        # Compute prediction and loss
        pred, _ = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()

        # Updates weights
        optimizer.step()

        if scheduler_step_flag:
            scheduler.step()
        
        if batch % 100 == 0 or batch == num_batches-1:
            loss = loss.item()
            current = batch * batch_size + len(X)
            current_lr = optimizer.param_groups[0]['lr']
            dim_size = len(str(size))
            output_string = f"\rEpoch {str(epoch+1).zfill(2)} - Loss: {loss:>2.5e}  [{current:>{dim_size}d}/{size:>{dim_size}d}]"
            output_string += f" - Learning Rate: {current_lr:>.4e}"
            print(output_string, end='', flush=True)
        
        if batch > 5:
            break
    
    if not scheduler_step_flag:
        scheduler.step()
    
    if not isinstance(loss, float):
        loss = loss.item()
    
    return loss
