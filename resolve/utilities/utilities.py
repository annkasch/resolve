#!/usr/bin/env python3
import os
import sys
from termcolor import colored
import numpy as np
import h5py
import random
import torch
import operator
import functools
import matplotlib.pyplot as plt
import gc

def set_random_seed(seed=42):
    random.seed(seed)           # Python's built-in random module
    np.random.seed(seed)        # NumPy random seed
    torch.manual_seed(seed)     # PyTorch random seed
    torch.cuda.manual_seed_all(seed)  # if using GPU

    # Ensures reproducibility on GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_all_files(path_to_files, ending='.csv'):
    """This function finds all file in a directory with a given ending

    Args:
        path_to_files: files path
        ending: define ending of the files

    Returns:
        res: a list with all filenames
    """

    dir_path = './'

    filename = ""
    index=[i+1 for i in range(len(path_to_files)) if  path_to_files[i]=='/']

    if len(index)>0:
        dir_path=path_to_files[:index[-1]]
        filename=path_to_files[index[-1]:]

    res = []

    filelist=os.listdir(dir_path)
    filelist.sort()
    #filelist = sorted(filelist, key=int)
    for file in filelist:

        if file.startswith(filename) and file.endswith(ending):
            res.append(f'{dir_path}{file}')
    if len(res) == 0:
        print(f"Warning: No files found at {path_to_files}.")
    return res

def parse_condition(condition_str):
    ops = {
                '==': operator.eq,
                '!=': operator.ne,
                '>=': operator.ge,
                '<=': operator.le,
                '>': operator.gt,
                '<': operator.lt
            }

    for op in ops:
        if op in condition_str:
            value_str = condition_str.split(op)[1].strip()
            break
    value = float(value_str) if '.' in value_str else int(value_str)
    
    def _compare(x, op, value):
        """Top-level helper that is picklable."""
        return op(x, value)
    
    return functools.partial(_compare, op=ops[op], value=value)

def train_test_groupwise_split(
    *arrays,
    groups,
    test_size=0.2,
    seed=None,
    return_indices=False,
    device=None,
    ensure_min_per_group=1,
):
    g = torch.as_tensor(groups, device=device)
    N = g.shape[0]
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    # unique groups and inverse map
    if g.ndim == 1:
        uniq, inv, counts = torch.unique(g, return_inverse=True, return_counts=True)
    else:
        uniq, inv, counts = torch.unique(g, dim=0, return_inverse=True, return_counts=True)

    G = uniq.shape[0]

    order = torch.argsort(inv)
    starts = torch.zeros(G + 1, dtype=torch.long, device=device)
    starts[1:] = torch.cumsum(counts, dim=0)

    # consistent clamp arguments
    k = (counts.float() * test_size).round().to(torch.long)
    k = torch.max(k, torch.full_like(k, ensure_min_per_group))
    k = torch.min(k, counts)

    test_chunks = []
    for i in range(G):
        s, e = starts[i].item(), starts[i + 1].item()
        c = e - s
        if k[i] == c:
            test_chunks.append(order[s:e])
        else:
            perm = torch.randperm(c, generator=gen, device=device)[:k[i]]
            test_chunks.append(order[s + perm])

    test_idx = torch.sort(torch.cat(test_chunks))[0]
    keep = torch.ones(N, dtype=torch.bool, device=device)
    keep[test_idx] = False
    train_idx = torch.nonzero(keep, as_tuple=True)[0]

    if not arrays:
        arrays = (g,)

    out = []
    for arr in arrays:
        t = torch.as_tensor(arr, device=device)
        out.extend((t[train_idx], t[test_idx]))

    if return_indices:
        out.extend((train_idx, test_idx))
    return tuple(out)

def cleanup_workspace(locals_dict):
    """Clean up workspace by closing objects and freeing memory"""
    # Close objects that have close method
    for key in list(locals_dict.keys()):
        if hasattr(locals_dict[key], 'close'):
            locals_dict[key].close()
        locals_dict[key] = None
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_nested(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d

def find_selected_indices(hdf5_file, label_dict):
        """
        Gives back indices of only the specified columns from parameter_key inHDF5 file.
        label_dict is a dictionary with following structure {'key': "...",'label_key': "....",'selected_labels': ["radius","thickness",...]}
        """
        with h5py.File(hdf5_file, "r") as hdf:
            if label_dict['label_key'] not in hdf:
                selected_indices= [0] if len(hdf[label_dict['key']].shape) == 1 else list(range(hdf[label_dict['key']].shape[1]))
            else:
                selected_labels=label_dict['selected_labels']
                if selected_labels == None: 
                    selected_labels=hdf[label_dict['label_key']][:]

                labels = list(map(lambda x: x.decode() if isinstance(x, bytes) else x, hdf[label_dict['label_key']][:]))
                # Find indices of required columns **before** reading phi
                selected_indices = [labels.index(label) for label in selected_labels if label in labels]

            if not selected_indices:
                raise ValueError(f"None of the requested labels {selected_labels} exist in {label_dict['key']}!")
            return sorted(selected_indices)

def parse_slice_string(command):
    # Example input: "columns[:500]"
    
    # Strip the prefix and get the inside of the brackets
    if not command.startswith("columns[") or not command.endswith("]"):
        raise ValueError("Invalid format")

    slice_str = command[len("columns["):-1]  # Extract between brackets, e.g., ":500"
    parts = slice_str.split(':')

    # Convert each part to int or None if empty
    def parse_part(part):
        return int(part) if part.strip() != '' else 0

    start = parse_part(parts[0]) if len(parts) > 0 and parse_part(parts[0]) > 0 else 0
    end = parse_part(parts[1]) if len(parts) > 0 and parse_part(parts[1])>0 else -1
    return [start, end]

def get_feature_and_label_size(config_file):
    x_size = len(config_file["simulation_settings"]["theta_labels"]+config_file["simulation_settings"]["phi_labels"])
    name_y =config_file["simulation_settings"]["target_labels"]

    if isinstance(name_y,str):
        if name_y:
            tmp=parse_slice_string(name_y)
            if tmp[1] < 1 :
                raise ValueError('Missing target size (y_size): please specify it in the target label name using the format "columns[:y_size]" in your settings file.')
            y_size=tmp[1]-tmp[0]
        else:
            y_size = 1
    else:
        y_size = len(name_y)
    
    return [x_size, y_size]

def plot(prediction_y_training, target_y_training, it=None, target_range=[0,1], sub_title=""):
        # Handle both 1D and 2D inputs
        if prediction_y_training.ndim == 1:
            prediction_y_training = prediction_y_training[:, np.newaxis]
        if target_y_training.ndim == 1:
            target_y_training = target_y_training[:, np.newaxis]

        n_outputs = prediction_y_training.shape[1]

        # Create mask for signal (assumes binary classification where signal ~1)
        if target_range[1] == 1:
            mask_signal = target_y_training > 0.5
        else:
            mask_signal = np.full_like(target_y_training, False, dtype=bool)

        mask_bkg = ~mask_signal

        # Split predictions and targets by signal/background
        prediction_signal_training = prediction_y_training[mask_signal].reshape(-1, n_outputs)
        target_signal_training = target_y_training[mask_signal].reshape(-1, n_outputs)
        prediction_bkg_training = prediction_y_training[mask_bkg].reshape(-1, n_outputs)
        target_bkg_training = target_y_training[mask_bkg].reshape(-1, n_outputs)

        fig, ax = plt.subplots(ncols=n_outputs, figsize=(5 * n_outputs, 4))

        if n_outputs == 1:
            ax = [ax]

        if it is not None:
            fig.suptitle(f'Iteration {it}', fontsize=10)

        bins = 100

        for k in range(n_outputs):
            has_signal = target_signal_training.shape[0] > 0

            if has_signal:
                ax[k].hist(target_signal_training[:, k], range=target_range, bins=bins, color='orangered', alpha=1.0, label='label (signal)')

            ax[k].hist(target_bkg_training[:, k], range=target_range, bins=bins, color=(3/255, 37/255, 46/255), alpha=0.8, label='label (bkg)')
            ax[k].hist(prediction_bkg_training[:, k], range=target_range, bins=bins, color=(113/255, 150/255, 159/255), alpha=0.8, label='network (bkg)')

            if has_signal:
                ax[k].hist(prediction_signal_training[:, k], range=target_range, bins=bins, color='coral', alpha=0.8, label='network (signal)')

            ax[k].set_yscale('log')
            ax[k].set_ylim(1e-1,1e6)
            ax[k].set_ylabel("Count")
            ax[k].set_xlabel("score")
            ax[k].set_title(sub_title, fontsize=10)

            fig.subplots_adjust(bottom=0.3, wspace=0.33)

            if has_signal:
                ax[k].legend(
                    labels=['true (signal)', 'label (bkg)', 'prediction (bkg)', 'prediction (signal)'],
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.2),
                    ncol=2,
                    frameon=False
                )
            else:
                ax[k].legend(
                    labels=['true', 'prediction'],
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.2),
                    ncol=2,
                    frameon=False
                )

        return fig

def make_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    else:
        return obj

def INFO(output):
    try:
        print(colored('[INFO] '+output, 'green'))
    except:
        print(colored('[INFO] '+str(output), 'green'))

def WARN(output):
    try:
        print(colored('[WARNING] '+output, 'yellow'))
    except:
        print(colored('[WARNING] '+str(output), 'yellow'))

def ERROR(output):
    try:
        print(colored('[ERROR] '+output, 'red'))
    except:
        print(colored('[ERROR] '+str(output), 'red'))
    sys.exit()