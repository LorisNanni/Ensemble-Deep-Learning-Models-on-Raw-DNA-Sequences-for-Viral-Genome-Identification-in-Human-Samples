import sys
import copy

from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

def DEBUG(x=None):
    out = f'[DEBUG {Path(__file__).name}]'
    if x: out += f' - {x}'
    print(out)

# ================= #
# SUPER MODEL CLASS #
# ================= #

class SuperModel(nn.Module):

    def __init__(self, model_infos):
        super().__init__()
        if isinstance(model_infos, str):
            self.load_model(Path(model_infos))
        elif isinstance(model_infos, Path):
            self.load_model(model_infos)
        else:
            self.model_params = copy.copy(model_infos['parameters'])
            self.normalization =  True if 'norm' in model_infos['training_options'] else False
            self._define_model()
            self.init_weigths = True if 'init' in model_infos['training_options'] else False
            if self.init_weigths:
                # Set the seed
                self.set_seed(39)       
                # Initialize weights
                self._initialize_weights()

        self.final_sigmoid = nn.Sigmoid()
        self.device = torch.device("cpu")

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # Set the seed for CUDA

    def _initialize_weights(self):
        raise NotImplementedError("This method must be implemented")

    def set_best_available_device(self, verbose=None):
        # Get the best available device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        if self.device != device: 
            self.device = device
            if verbose: print(f"\nUsing {self.device} device\n")
            self.to(self.device)
        elif verbose:
            print(f'Already set to best available device: {self.device}')

    def set_to_cpu(self):
        self.device = 'cpu'
        self.to('cpu')

    def _define_model(self):
        raise NotImplementedError("This method must be implemented")

    def forward(self, x):
        logit = self._inner_forward(x)
        pred = self.final_sigmoid(logit)        
        return pred, logit
        
    def _inner_forward(self, x):
        raise NotImplementedError("This method must be implemented")

    def get_model_infos(self):
        models_infos = {
            'parameters': self.model_params,
            'normalization': self.normalization,
            'init_weights': self.init_weigths
        }
        return models_infos

    def save_model(self, path):
        if self.device != 'cpu': self.to('cpu')
        saving_state = self.get_model_infos()
        saving_state['state_dict'] = self.state_dict()
        torch.save(saving_state, path)
        if self.device != 'cpu': self.to('cuda')
        return saving_state, path
    
    def load_model(self, path):
        saving_state = torch.load(path)       
        self.model_params = saving_state['parameters']
        self.normalization = saving_state['normalization']
        self.init_weigths = saving_state['init_weights']
        self._define_model()
        status = self.load_state_dict(saving_state['state_dict'], strict=True)
        print(f"Model loaded from {path} with status: {status}")
        self.eval()
    
    def is_equal(self, other):
        if not isinstance(other, self.__class__): return False 

        if str(self) != str(other): return False

        self_params = self.state_dict()
        other_params = other.state_dict()

        for key in self_params.keys():
            if key not in other_params: return False
            if not torch.equal(self_params[key], other_params[key]): return False
            
        return True

    def train_model(self, train_dl, loss_fn, optimizer, scheduler, epochs,
                    epsilon=0, min_score=0,
                    validation_fn=None, val_dl=None, verbose=0):
        
        self.verbose = verbose

        if validation_fn:
            score_collection = []
            best_model = None
            best_score = -1
            best_epoch = 0
        
        # Early stopping counters
        no_improvement = 0 # if no improvement of epsilon after 6 epochs
        no_over_min_score = 0 # if no score > min_score after 6 epochs

        for epoch in range(epochs):       
            show_device = self.__verbose_lv(1) if epoch == 0 else 0
            self.set_best_available_device(verbose=show_device)
            # Train
            # loss = self.__train_one_epoch(train_dl, loss_fn, optimizer, scheduler, epoch)
            with torch.autograd.set_detect_anomaly(True): loss = self.__train_one_epoch(train_dl, loss_fn, optimizer, scheduler, epoch)

            # Validate
            if validation_fn:
                score, _, _ = self.test_model(val_dl, validation_fn)
                
                if not sys.stdout.isatty() and self.__verbose_lv(2):
                    output_string = f"Epoch {str(epoch+1).zfill(2)} - Loss: {loss:>2.5e}"
                    output_string += f" - Validation Score: {score:>.5f}"
                    print(output_string)
                elif self.__verbose_lv(2):
                    print(f" - Validation Score: {score:>.5f}")
            
                # Save score for plot
                score_collection.append(score)
                
                # Check score improvement > epsilon
                if (score - best_score) > epsilon:
                    best_model = copy.deepcopy(self.state_dict())
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
                    if self.__verbose_lv(1):
                        print(f"Training stopped at epoch: #{str(epoch+1).zfill(2)} beacuse no Score improvement after 6 epochs (with error: {epsilon:>.5f})")
                    break    
                # If no improvemnt than min_score after 6 epochs   
                if no_over_min_score > 6:
                    if self.__verbose_lv(1):
                        print(f"Training stopped at epoch: #{str(epoch+1).zfill(2)} beacuse no Score >= {min_score:>.5f} after 6 epochs")
                    break

        if validation_fn:
            self.load_state_dict(best_model)
            self.set_to_cpu()
            
            if self.__verbose_lv(1):
                print(f"Best model at epoch: #{str(best_epoch).zfill(2)} with Validation Score = {best_score:>.5f}")
                print()
        
            return score_collection
        
        self.set_to_cpu()
        
    def __train_one_epoch(self, dataloader, loss_fn, optimizer, scheduler, epoch):
        size = len(dataloader.dataset)
        batch_size = dataloader.batch_size
        num_batches = len(dataloader)

        # If True, the step is taken at the end of each batch; otherwise at the end of the epoch
        scheduler_step_flag = isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

        # To detect gradient anomalies
        torch.autograd.set_detect_anomaly(True)

        # Set the model to training mode - important for batch normalization and dropout layers
        self.train()
        for batch, (X, y) in tqdm(enumerate(dataloader), total=num_batches, desc=f"Epoch {str(epoch+1).zfill(2)}"):
            
            # Set the gradients to zero
            optimizer.zero_grad()
            # Transfer data to correct device
            X = X.to(self.device)
            y = y.to(self.device)
            
            # Reshape labels vector from a row to a column
            if batch_size > 1:
                y = y.reshape(-1,1)

            # Compute prediction and loss
            pred, _ = self(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()

            # Updates weights
            optimizer.step()

            if scheduler_step_flag:
                scheduler.step()
            
            if sys.stdout.isatty() and self.__verbose_lv(2) and (batch % 100 == 0 or batch == num_batches-1) and sys.stdout.isatty():
                loss = loss.item()
                current = batch * batch_size + len(X)
                current_lr = optimizer.param_groups[0]['lr']
                dim_size = len(str(size))
                output_string = f"\rEpoch {str(epoch+1).zfill(2)} - Loss: {loss:>2.5e}  [{current:>{dim_size}d}/{size:>{dim_size}d}]"
                output_string += f" - Learning Rate: {current_lr:>.4e}"
                print(output_string, end='', flush=True)
        
        if not scheduler_step_flag:
            scheduler.step()
        
        if not isinstance(loss, float):
            loss = loss.item()
        
        return loss

    def test_model(self, dataloader, score_fn, verbose=None):
        self.set_best_available_device(verbose=verbose)

        # Set the model to evaluation mode - important for batch normalization and dropout layers
        self.eval()
        
        # For scores computation
        y_pred_prob = torch.empty(0).to(self.device)
        y_true = torch.empty(0).to(self.device)
        
        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for (X, y) in dataloader:
            
                # Transfer data to correct device
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Reshape labels vector from a row to a column
                if dataloader.batch_size > 1:
                    y = y.reshape(-1,1)
                
                pred, _ = self(X)
                
                y_pred_prob = torch.hstack((y_pred_prob,pred.flatten()))
                y_true = torch.hstack((y_true,y.flatten()))
        
        if self.device != 'cpu':
            y_true = y_true.to('cpu')
            y_pred_prob = y_pred_prob.to('cpu')
            
        self.set_to_cpu()

        return score_fn(y_true, y_pred_prob)
    
    def params_string(self):
        var_str =  f" - Init Weights = {self.init_weigths}"
        var_str += f"\n - Normalization = {self.normalization}"
        return var_str
    
    def __verbose_lv(self, lv):
        return self.verbose > lv-1
    
    def __compute_grad_norm(self):
        total_norm = 0.0
        count = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)  # L2 norm
                total_norm += param_norm.item()
                count += 1
        if count > 0:
            return total_norm / count
        else:
            return 0.0

# ========== #
# CNN Models #
# ========== #

# BRANCH CLASS

class Branch(SuperModel):

    def __init__(self, model_infos):
        super().__init__(model_infos)
        self.set_trainability(True)
    
    def _define_model(self):
        # model_infos
        sample_dim = self.model_params['sample_dim']
        features_size = self.model_params['features_size']
        filter_size = self.model_params['filter_size']
        kernel_size = self.model_params['kernel_size']
        dropout_prob = self.model_params['dropout_prob']
        pooling_layer = self.model_params['pooling_layer'] 
        norm_type = self.model_params['norm_type'] if 'norm_type' in self.model_params else None

        # Model Definition
        self.conv = nn.Conv1d(features_size, filter_size, kernel_size)
        self.batch_conv = nn.BatchNorm1d(filter_size)
        kernel_size_pooling = sample_dim-kernel_size+1
        if norm_type:
            # Power Avg Pooling case
            self.pooling = pooling_layer(norm_type,kernel_size_pooling)
        else:
            # Max or Avg Pooling case
            self.pooling = pooling_layer(kernel_size_pooling)
        self.dropout_pool = nn.Dropout(dropout_prob)
        self.relu_conv = nn.ReLU()
        self.flatten = nn.Flatten(-2,-1)
        self.fc1 = nn.Linear(filter_size, filter_size)
        self.batch_fc1 = nn.BatchNorm1d(filter_size)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(filter_size, 1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He for ReLU
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Set bias to 0 

    def _inner_forward(self, x):
        # Convolutional Layer
        x = self.conv(x)
        if self.normalization: x = self.batch_conv(x)
        x = self.relu_conv(x)
        
        # Pooling Layer, one pooling for each filter
        if type(self.pooling) == nn.LPPool1d:
            x = torch.clamp(x, min=1e-3, max=1e3)
        x = self.pooling(x)
        x = self.dropout_pool(x)
        
        # From column to row vector
        x = self.flatten(x)
        
        # Fully-Connected Layer
        x = self.fc1(x)
        if self.normalization: x = self.batch_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)
        
        # Output Layer
        x = self.fc2(x)

        return x # x= Logits

    def set_trainability(self, trainable):
        self.model_params['trainable'] = trainable
        for param in self.parameters():
            param.requires_grad = trainable

    def get_model_infos(self):
        models_infos = {
            'model_type': 'Branch'
        }
        models_infos.update(super().get_model_infos())
        return models_infos

    def params_string(self):
        var_str =  f" - Filter Size = {str(self.model_params['filter_size']).zfill(4)}"
        var_str += f"\n - Kernel Size = {str(self.model_params['kernel_size']).zfill(2)}"
        var_str += f"\n - Dropout Prob. = {self.model_params['dropout_prob']}"
        var_str += f"\n - Pooling Layer = "
        if self.model_params['pooling_layer'] == torch.nn.AvgPool1d: var_str += "AvgPool"
        elif self.model_params['pooling_layer'] == torch.nn.MaxPool1d: var_str += "MaxPool"
        else: var_str += f"LPPool, Norm Type = {self.model_params['norm_type']}"
        var_str += f"\n - Trainable = {self.model_params['trainable']}"
        var_str += "\n" + super().params_string()
        return var_str
        
# MERGE CLASS

class Merger(SuperModel):
    
    def __init__(self, model_infos):
        super().__init__(model_infos)
    
    def _define_model(self):
        # model_infos
        branches = self.model_params['branches_dict']
        dropout_prob = self.model_params['dropout_prob']

        # Model Definition
        # Branches Definition
        self.branches = nn.ModuleDict()
        output_size = 0
        self.model_params['branches_dict'] = {}
        # "Deleting" the last 3 layers of every single branch
        for name, model in branches.items():
            if isinstance(model, str) or isinstance(model, Path):
                model = Branch(model)
                trainable = False
            else: # Model is a Branch object
                trainable = True
            output_size += model.fc1.out_features
            model_infos = dict()
            model_infos['parameters'] = copy.deepcopy(model.model_params)
            state_dict = copy.deepcopy(model.state_dict())
            state_dict.pop('fc2.weight');
            state_dict.pop('fc2.bias');
            training_options = 'no-inputs'
            if model.init_weigths and model.normalization: training_options = 'init+norm'
            elif model.normalization: training_options = 'norm'
            elif model_infos: training_options = 'init-weights'
            model_infos['training_options'] = training_options
            branch = self.Branch(model_infos)
            branch.set_trainability(trainable)
            branch.load_state_dict(state_dict)
            branch.eval()
            self.branches[name] = branch
            self.model_params['branches_dict'][name] = copy.copy(model_infos)
        
        # Dimension of union of all branches output
        self.model_params['output_size'] = output_size

        # Adding last 3 layers: Dropout -> FC -> Sigmoid
        self.dropout = nn.Dropout(dropout_prob)
        self.fc_final = nn.Linear(output_size,1)
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc_final.weight)
        nn.init.constant_(self.fc_final.bias, 0)

    def _inner_forward(self, x):
        branch_results = torch.empty(0).to(self.device)
        for branch_name, branch in self.branches.items():
            # Running singular Branches
            result = branch(x)
            # Concatenate
            branch_results = torch.hstack([branch_results,result])

        branch_results = self.dropout(branch_results)
        output = self.fc_final(branch_results)
        return output
    
    def load_model(self, path):
        saving_state = torch.load(path, weights_only=False)            
        self.model_params = saving_state['parameters']
        branches = self.model_params['branches_dict']
        
        # Branches Definition
        self.branches = nn.ModuleDict()
        for name, model_params in branches.items():
            branch = self.Branch(model_params)
            if "trainable" in model_params:
                branch.set_trainability(model_params['trainable'])
            else:
                # print("WARNING: No trainability info for branch '{name}' in the loaded model, setting it to False")
                branch.set_trainability(False)
            self.branches[name] = branch

        # Adding last 3 layers: Dropout -> FC -> Sigmoid
        self.dropout = nn.Dropout(self.model_params['dropout_prob'])
        self.fc_final = nn.Linear(self.model_params['output_size'],1)

        status = self.load_state_dict(saving_state['state_dict'], strict=True)
        print(f"[i] Model loaded from {path} with status: {status}")
        self.eval()

    def set_branches_trainability(self, trainable):
        for branch in self.branches.values():
            branch.set_trainability(trainable)

    def get_model_infos(self):
        models_infos = {
            'model_type': 'Merger'
        }
        models_infos.update(super().get_model_infos())
        return models_infos

    def params_string(self):
        var_str =  f" - Dropout Prob. = {self.model_params['dropout_prob']}"
        super_str = super().params_string()
        var_str += f'\n{super_str}'
        var_str += f"\n + Branches Infos:"
        for name, model in self.branches.items():
            var_str += f'\n +- {name}:'
            model_str = model.params_string().replace('-','+---')
            var_str += f'\n{model_str}'

        return var_str

    # Inner Branch Class (Branch class without the last 2 layers)
    class Branch(Branch):
        def __init__(self, model_infos):
            super().__init__(model_infos)
            self.set_trainability(False)
            
        def _define_model(self):
            # model_infos
            sample_dim = self.model_params['sample_dim']
            features_size = self.model_params['features_size']
            filter_size = self.model_params['filter_size']
            kernel_size = self.model_params['kernel_size']
            dropout_prob = self.model_params['dropout_prob']
            pooling_layer = self.model_params['pooling_layer'] 
            norm_type = self.model_params['norm_type'] if 'norm_type' in self.model_params else None

            # Model Definition
            
            # Convolutional Layer copy
            self.conv = nn.Conv1d(features_size, filter_size, kernel_size)
            self.batch_conv = nn.BatchNorm1d(filter_size)

            kernel_size_pooling = sample_dim-kernel_size+1
            if norm_type:
                # Power Avg Pooling case
                self.pooling = pooling_layer(norm_type,kernel_size_pooling)
            else:
                # Max or Avg Pooling case
                self.pooling = pooling_layer(kernel_size_pooling)
            self.dropout_pool = nn.Dropout(dropout_prob)
            self.relu_conv = nn.ReLU()
            self.flatten = nn.Flatten(-2,-1)

            # Linear Layer copy
            self.fc1 = nn.Linear(filter_size, filter_size)
            self.batch_fc1 = nn.BatchNorm1d(filter_size)

            self.relu_fc1 = nn.ReLU()
        
        def forward(self, x):
            result =  self._inner_forward(x)
            return result

        def _inner_forward(self, x):
            # Convolutional Layer
            x = self.conv(x)
            if self.normalization: x = self.batch_conv(x)
            x = self.relu_conv(x)
            # Pooling Layer, one pooling for each filter
            x = self.pooling(x)
            x = self.dropout_pool(x)
            # From column to row vector
            x = self.flatten(x)
            # Fully-Connected Layer
            x = self.fc1(x)
            if self.normalization: x = self.batch_fc1(x)
            x = self.relu_fc1(x)
            return x