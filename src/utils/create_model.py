from src.model.network import Merger, Branch
import torch
import torch.nn as nn
from pathlib import Path



def create_model(model_infos):
    if isinstance(model_infos, str) or isinstance(model_infos, Path):
        model_type = eval(torch.load(model_infos)['model_type'])
    elif 'branches_dict' in model_infos['parameters']:
        model_type = Merger
    else:
        model_type = Branch
        if isinstance( model_infos['parameters']['pooling_layer'],str):
            model_infos['parameters']['pooling_layer'] = eval(model_infos['parameters']['pooling_layer'])
    
    return model_type(model_infos)