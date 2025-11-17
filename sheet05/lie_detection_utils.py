import torch as t
import pandas as pd
import numpy as np
import os
from glob import glob
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTS_BATCH_SIZE = 25

def collect_acts(dataset_name, model_family, model_size,
                  model_type, layer, path_to_acts=None, center=True, scale=False, device='cpu'):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """
    if path_to_acts is None:
        path_to_acts = os.path.join(ROOT, 'acts')
    directory = os.path.join(path_to_acts, model_family, model_size, model_type, dataset_name)
    activation_files = glob(os.path.join(directory, f'layer_{layer}_*.pt'))
    acts = [t.load(os.path.join(directory, f'layer_{layer}_{i}.pt'), map_location=device) for i in range(0, ACTS_BATCH_SIZE * len(activation_files), ACTS_BATCH_SIZE)] 
    acts = t.cat(acts, dim=0).to(device)
    if center:
        acts = acts - t.mean(acts, dim=0)
    if scale:
        acts = acts / t.std(acts, dim=0)
    return acts

def cat_data(d):
    """
    Given a dict of datasets (possible recursively nested), returns the concatenated activations and labels.
    """
    all_acts, all_labels = [], []
    for dataset in d:
        if isinstance(d[dataset], dict):
            if len(d[dataset]) != 0: # disregard empty dicts
                acts, labels = cat_data(d[dataset])
                all_acts.append(acts), all_labels.append(labels)
        else:
            acts, labels = d[dataset]
            all_acts.append(acts), all_labels.append(labels)
    return t.cat(all_acts, dim=0), t.cat(all_labels, dim=0)

class DataManager:
    """
    Class for storing activations and labels from datasets of statements.
    """
    def __init__(self):
        self.data = {
            'train' : {},
            'val' : {}
        } # dictionary of datasets
        self.proj = None # projection matrix for dimensionality reduction
    
    def add_dataset(self, dataset_name, model_family, model_size, model_type, layer, path_to_datasets=None, path_to_acts=None,
                     label='label', split=None, seed=None, center=True, scale=False, device='cpu'):
        """
        Add a dataset to the DataManager.
        label : which column of the csv file to use as the labels.
        If split is not None, gives the train/val split proportion. Uses seed for reproducibility.
        """
        acts = collect_acts(dataset_name, model_family, model_size, model_type,
                             layer, center=center, scale=scale, device=device, path_to_acts=path_to_acts)
        if path_to_datasets is None:
            path_to_datasets = os.path.join(ROOT, 'datasets')
        df = pd.read_csv(os.path.join(path_to_datasets, f'{dataset_name}.csv'))
        labels = t.Tensor(df[label].values).to(device)

        if split is None:
            self.data[dataset_name] = acts, labels

        if split is not None:
            assert 0 <= split and split <= 1
            if seed is None:
                seed = random.randint(0, 1000)
            t.manual_seed(seed)
            train = t.randperm(len(df)) < int(split * len(df))
            val = ~train
            self.data['train'][dataset_name] = acts[train], labels[train]
            self.data['val'][dataset_name] = acts[val], labels[val]

    def get(self, datasets):
        """
        Output the concatenated activations and labels for the specified datasets.
        datasets : can be 'all', 'train', 'val', a list of dataset names, or a single dataset name.
        If proj, projects the activations using the projection matrix.
        """
        if datasets == 'all':
            data_dict = self.data
        elif datasets == 'train':
            data_dict = self.data['train']
        elif datasets == 'val':
            data_dict = self.data['val']
        elif isinstance(datasets, list):
            data_dict = {}
            for dataset in datasets:
                if dataset[-6:] == ".train":
                    data_dict[dataset] = self.data['train'][dataset[:-6]]
                elif dataset[-4:] == ".val":
                    data_dict[dataset] = self.data['val'][dataset[:-4]]
                else:
                    data_dict[dataset] = self.data[dataset]
        elif isinstance(datasets, str):
            data_dict = {datasets : self.data[datasets]}
        else:
            raise ValueError(f"datasets must be 'all', 'train', 'val', a list of dataset names, or a single dataset name, not {datasets}")
        acts, labels = cat_data(data_dict)
        # if proj and self.proj is not None:
        #     acts = t.mm(acts, self.proj)
        return acts, labels


     




