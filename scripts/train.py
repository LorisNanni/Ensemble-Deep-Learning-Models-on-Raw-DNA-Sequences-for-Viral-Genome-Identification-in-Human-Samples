

from pathlib import Path
import os
import torch
import torch.nn as nn
import json, copy
import argparse
from src.dataset.csv_dataset import CSVDataset
from sklearn.model_selection import ParameterGrid
from src.utils.metrics import auroc as score_fn
from src.utils.create_model import create_model
from src.utils.train_model import train_model
from src.utils.test_model import test_model

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", "-d", default="data", help="Path to the folder containing csv files.")
    parser.add_argument("--save_dir", "-s", default="savefolder", help="Directory where the trained models and results will be saved.")
    parser.add_argument("--checkpoint", "-c",  default=None, help="Path to the model checkpoint to load (if not provided, training will start from scratch).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output during testing.")

    parser.add_argument("--path", "-p", required=True, help="Path to the '.json' file with training parameters")

    return parser.parse_args()


def start_training(model_infos, dl_train, dl_valid):

    def __set_to_list(obj):
        # if not isinstance(obj, list): return [obj]
        # else: return obj
        return obj if isinstance(obj,list) else [obj]
    
    def __params_to_str(hyperparameters, model):
        param_str = "Hyperparameters:"
        param_str += f"\n - Batch Size = {hyperparameters['batch_size']}"
        param_str += f"\n - Learning Rate = {hyperparameters['learning_rate']}"
        param_str += f"\n - Epochs = {hyperparameters['epochs']}"
        param_str += "\nModel Parameters:"
        model_str = model.params_string()
        param_str += f"\n{model_str}"
        return param_str

    hyperparameters = model_infos['hyperparameters']
    parameters = model_infos['parameters']

    # Prepare hyperparameters
    for k,v in hyperparameters.items():
        hyperparameters[k] = __set_to_list(v)
    h_grid = {'hyperparameters': list(ParameterGrid(hyperparameters))}

    # Prepare parameters
    for k,v in parameters.items():
        parameters[k] = __set_to_list(v)        
    p_grid = {'parameters': list(ParameterGrid(parameters))}
    
    # Create the grid from parameters and hyperparameters
    grid = ParameterGrid({**h_grid,**p_grid})

    best_score = 0

    print(f'Number of models to cross validate = {len(grid)}\n')

    dim_grid = len(grid)
    models_count = 0
    best_count = 0

    best_model_infos = {}

    for el in grid:
        models_count += 1

        lr = el['hyperparameters']['learning_rate']
        epochs = el['hyperparameters']['epochs']

        epochs = el['hyperparameters']['epochs'] = 1
        el['hyperparameters']['batch_size'] = 1024

        model = create_model({**model_infos, 'parameters': el['parameters']})
        
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        print(f"Start Training: Model {str(models_count).zfill(len(str(dim_grid)))}/{dim_grid}", end='')
        print(f"\n\n{__params_to_str(el['hyperparameters'], model)}")
        print()
    
        # If no init min score is set, every training start with a min_score=0
        if model_infos['min_score'] < 0:
            min_score = 0
        # If init min score is > best score, the training min score is set from the value that has been passed as argument
        elif model_infos['min_score'] > best_score:
            min_score = model_infos['min_score']
        # If the best score is > the init min score, the training min score is set with the best score computed so far 
        else:
            min_score = best_score
        
        validation_scores = train_model(model, train_dl=dl_train, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, epochs=epochs,
                                            epsilon=model_infos['epsilon'], min_score=min_score,
                                            validation_fn=score_fn, val_dl=dl_valid, verbose=2)
        
        # if not sys.stdout.isatty() and model_infos['show_graphs']: self.__graph_validation_score(validation_scores)

        candidate_best_score = max(validation_scores)

        if candidate_best_score > best_score:
            best_score = candidate_best_score
            best_model = copy.deepcopy(model)
            best_hyperparameters = el['hyperparameters']
            best_count = models_count
            best_model_infos = {
                **model.get_model_infos(),
                'hyperparameters': el['hyperparameters'],
                'val_score_per_epoch': validation_scores,
                'val_score': best_score                    
            }

    result_string = f'Best learned model is #{str(best_count).zfill(2)} with '
    result_string += f'AUROC Best Validation Score = {best_score:>.5f} and:'
    result_string += f'\n{__params_to_str(best_hyperparameters, best_model)}'
    result_string += '\n'
    print(result_string)

    return best_model, best_model_infos


def main():
    args = parse_args()

    print("[i] Starting training with the following parameters:")
    print(" - data folder:", args.data_folder)
    print(" - checkpoint:", args.checkpoint)
    print(" - save directory:", args.save_dir)

    train_file = os.path.join(args.data_folder, "fullset_train.csv")
    valid_file = os.path.join(args.data_folder, "fullset_validation.csv")
    test_file  = os.path.join(args.data_folder, "fullset_test.csv")
    assert os.path.isfile(train_file), f"Training data file '{train_file}' does not exist."
    assert os.path.isfile(valid_file), f"Validation data file '{valid_file}' does not exist."
    assert os.path.isfile(test_file), f"Test data file '{test_file}' does not exist."

    ## assert no relative path. Otherwise saving mechanism fails
    assert not args.path.startswith('.'), "Relative paths are not allowed for saving. Please provide an absolute path in the 'saving_name' field of the model info json file."
    

    os.makedirs(args.save_dir, exist_ok=True)

    assert Path(args.path).is_file()
    models_collection = json.loads(Path(f'{args.path}').read_text())
    if not isinstance(models_collection, list): models_collection = [models_collection]
    models_collection[0]['src_path'] = str(args.path)


    tot_models = len(models_collection)
    print(f'Number of models to learn: {tot_models}\n')
    count = 0

    for model_infos in models_collection:
        # Show some infos related to learning
        count += 1
        print(f'\n{str(count).zfill(len(str(tot_models)))}/{tot_models} model in learning phase')
        print(f'Name: {model_infos["name"]}')
        print(f'Training Options: {model_infos["training_options"]}')
        if 'bidirectional' not in model_infos['parameters']: print(f'Encoding function: {model_infos["encoding_fn"]}')
        print(f'Hyperparameters: {model_infos["hyperparameters"]}')
        print(f'Parameters: {model_infos["parameters"]}')

        batch_size = model_infos['hyperparameters']['batch_size'] # if 'batch_size' in model_infos['hyperparameters'] else 128
        print("[i] using batch size =", batch_size)

        train_df = CSVDataset(f"{train_file}", transform="onehot_encoding")
        valid_df = CSVDataset(f"{valid_file}", transform="onehot_encoding")
        test_df  = CSVDataset(f"{test_file}", transform="onehot_encoding")
        dl_train = train_df.get_dataloader(batch_size=batch_size, shuffle=True, drop_last=False)
        dl_valid = valid_df.get_dataloader(batch_size=batch_size, shuffle=False, drop_last=False)
        dl_test  = test_df.get_dataloader(batch_size=batch_size, shuffle=False, drop_last=False)
        print("[i] found ", len(train_df), "samples in the train dataset.")
        print("[i] found ", len(valid_df), "samples in the valid dataset.")
        print("[i] found ", len(test_df), "samples in the test dataset.")

        # Set the correct branch model path for merger models
        if 'branches_dict' in model_infos['parameters'] and 'not_final' in args.path:
            for k,v in model_infos['parameters']['branches_dict'].items():
                print("adapting")
                v = v.replace('.ptm','')
                v = f'{os.path.dirname(args.path)}/final/{model_infos["encoding_fn"].replace("_encoding","")}/branch/{v}/{model_infos["training_options"]}/model.ptm'
                model_infos['parameters']['branches_dict'][k] = v

        # Train model
        model, result_infos = start_training(model_infos, dl_train, dl_valid)
        # updating the result infos with the model infos (name, encoding function, training options, saving name) to save all the data in a single json file
        result_infos = {
            'name': model_infos['name'],
            'encoding_fn': model_infos['encoding_fn'],
            'training_options' : model_infos['training_options'],
            'saving_name': model_infos['saving_name'],
            **result_infos
        }

        # Final test 
        auroc, fpr, tpr, = test_model(model, dl_test, score_fn)
        result_infos['test'] = {
            'score': auroc,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }

        result_infos['parameters'] = copy.copy(result_infos['parameters'])

        print('\nSAVING PATHS: \n')

        # main_path = model_infos['src_path'].replace('.json', '')
        # main_path = main_path.replace('ready_to_train_files/', '')



        # Save Model
        if model_infos['save_model']:
            path = Path(f'{args.save_dir}/model.ptm')
            path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(path)
            print(f' - Model Saved in:\n\t{path}')
        
        # Sava results data: - model infos - validation score - test score
        path = Path(f'{args.save_dir}/data.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        if 'pooling_layer' in result_infos['parameters']:
            result_infos['parameters']['pooling_layer'] = model_infos['parameters']['pooling_layer'][0]
        elif 'branches_dict' in result_infos['parameters']:
            result_infos['parameters']['branches_dict'] = model_infos['parameters']['branches_dict'][0]
        path.write_text(json.dumps(result_infos, indent=4))
        print(f' - Training Results Data Saved in:\n\t{path}')
        print()


if __name__ == "__main__":
    main()