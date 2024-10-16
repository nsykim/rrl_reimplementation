import os
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Save model checkpoint.
    
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): The current epoch.
        filepath (str): The path to save the checkpoint.
    """
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filepath (str): The path to the checkpoint.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into.
    
    Returns:
        int: The epoch the checkpoint was saved at.
    """
    state = torch.load(filepath)
    model.load_state_dict(state['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    return state['epoch']

def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    
    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    return metrics

def save_json(data, filepath):
    """
    Save data to a JSON file.
    
    Args:
        data (dict): The data to save.
        filepath (str): The path to the JSON file.
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    """
    Load data from a JSON file.
    
    Args:
        filepath (str): The path to the JSON file.
    
    Returns:
        dict: The loaded data.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): The seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)