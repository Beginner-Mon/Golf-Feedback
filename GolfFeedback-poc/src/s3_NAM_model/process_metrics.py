# plot_nam_features.py - FIXED VERSION

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from nam.models import NAM, get_num_units
from nam.trainer import LitNAM
from nam.data import NAMDataset
from nam.config import Config, defaults
import yaml
from sklearn.preprocessing import MinMaxScaler

def normalize_feature_values(raw_values: np.ndarray, all_raw_data: np.ndarray) -> np.ndarray:
    """
    Normalize feature values using MinMaxScaler(-1, 1) like in transform_data()
    
    Args:
        raw_values: unique values cần normalize
        all_raw_data: toàn bộ data của feature đó để fit scaler
    
    Returns:
        normalized values in range [-1, 1]
    """
    scaler = MinMaxScaler((-1, 1))
    scaler.fit(all_raw_data.reshape(-1, 1))
    normalized = scaler.transform(raw_values.reshape(-1, 1)).flatten()
    return normalized


def get_feature_contributions(model: torch.nn.Module, dataset) -> dict:
    """
    Get contributions of each feature for its unique values.
    Returns dict: feature_name -> {'contributions', 'original_values', 'normalized_values'}
    """
    feat_contrib_dict = {}
    
    for i, feature_name in enumerate(dataset.features_names):
        # Lấy unique values GỐC
        original_values = np.array(sorted(dataset.raw_X[feature_name].unique()))
        
        # Lấy toàn bộ data GỐC của feature này
        all_raw_data = dataset.raw_X[feature_name].to_numpy()
        
        # Normalize giống như trong transform_data()
        normalized_values = normalize_feature_values(original_values, all_raw_data)
        
        # Pass normalized values vào model
        feature_values_tensor = torch.tensor(normalized_values).float().to(model.config.device)
        feat_contrib = model.feature_nns[i](feature_values_tensor).cpu().detach().numpy().squeeze()
        
        feat_contrib_dict[feature_name] = {
            'contributions': feat_contrib,
            'original_values': original_values,
            'normalized_values': normalized_values
        }
    
    return feat_contrib_dict


def get_ideal_feature_value(f_i: np.ndarray,
                            x_values: np.ndarray,
                            target: str = 'maximize') -> Tuple[float, float]:
    """
    Return ideal feature value x_i that maximizes or minimizes f_i
    """
    if target == 'maximize':
        idx_opt = np.argmax(f_i)
    elif target == 'minimize':
        idx_opt = np.argmin(f_i)
    else:
        raise ValueError("target must be 'maximize' or 'minimize'")

    return x_values[idx_opt], f_i[idx_opt]


def plot_nam_feature(model: torch.nn.Module,
                     dataset,
                     feature_name: str,
                     target: str = 'maximize',
                     n_blocks: int = 20,
                     line_color: str = 'black'):
    """
    Plot NAM function for a single numeric feature with density shading
    and highlight red line = ideal value according to target.
    Returns ideal value x_opt (in ORIGINAL scale).
    """
    # Check numeric feature
    if not np.issubdtype(type(dataset.ufo[feature_name][0]), np.number):
        print(f"Skipping non-numeric feature: {feature_name}")
        return None

    # Get contributions (ĐÃ NORMALIZED ĐÚNG)
    feat_data_contrib = get_feature_contributions(model, dataset)
    f_i = feat_data_contrib[feature_name]['contributions']
    x_i = feat_data_contrib[feature_name]['original_values']  # Dùng original values cho x-axis

    # Ideal feature value (trong original scale)
    x_opt, f_opt = get_ideal_feature_value(f_i, x_i, target=target)

    # Plot setup
    fig, ax = plt.subplots(figsize=(8,6))

    # Background density shading (dùng RAW data)
    single_feature_data = dataset.raw_X[feature_name].to_numpy()
    x_min, x_max = x_i.min(), x_i.max()
    segments = np.linspace(x_min, x_max, n_blocks+1)
    hist, _ = np.histogram(single_feature_data, bins=segments)
    hist = hist / hist.max()  # normalize
    y_min, y_max = f_i.min(), f_i.max()
    y_range = y_max - y_min
    for i in range(n_blocks):
        rect = patches.Rectangle(
            (segments[i], y_min - 0.1*y_range),
            segments[i+1]-segments[i],
            y_range*1.2,
            facecolor='gray',
            alpha=hist[i]*0.5
        )
        ax.add_patch(rect)

    # Plot NAM function line
    ax.plot(x_i, f_i, linestyle='--', color=line_color, linewidth=2)

    # Red line = ideal value
    ax.axvline(x=x_opt, color='red', linestyle='-', linewidth=2)
    ax.text(x_opt, y_max, f"Ideal={x_opt:.2f}", color='red', fontsize=12, rotation=90, va='bottom')

    # Labels
    ax.set_xlabel(feature_name, fontsize=14)
    ax.set_ylabel(f'f({feature_name})', fontsize=14)
    ax.set_title(f'NAM Function for {feature_name} (target={target})', fontsize=16)
    plt.show()

    return x_opt


def get_ideal_values(model: torch.nn.Module,
                     dataset,
                     feature_list: List[str] = None,
                     target: str = 'maximize') -> dict:
    """
    Compute ideal values for numeric features according to the NAM model.
    Returns dict: feature_name -> ideal value (in ORIGINAL scale)
    """
    if feature_list is None:
        feature_list = dataset.features_names

    # Filter numeric features
    numeric_features = [f for f in feature_list if np.issubdtype(type(dataset.ufo[f][0]), np.number)]
    if not numeric_features:
        print("No numeric features found.")
        return {}

    feat_data_contrib = get_feature_contributions(model, dataset)
    ideal_values = {}

    for feature_name in numeric_features:
        f_i = feat_data_contrib[feature_name]['contributions']
        x_i = feat_data_contrib[feature_name]['original_values']
        x_opt, _ = get_ideal_feature_value(f_i, x_i, target=target)
        ideal_values[feature_name] = x_opt

    return ideal_values


def plot_multiple_nam_features(model: torch.nn.Module,
                               dataset,
                               feature_list: List[str] = None,
                               target: str = 'maximize',
                               n_cols: int = 2,
                               n_blocks: int = 20) -> dict:
    """
    Plot multiple numeric features in subplots with NAM function style.
    Returns dict: feature_name -> ideal value (in ORIGINAL scale)
    """
    if feature_list is None:
        feature_list = dataset.features_names

    # Filter numeric features
    numeric_features = [f for f in feature_list if np.issubdtype(type(dataset.ufo[f][0]), np.number)]
    if not numeric_features:
        print("No numeric features to plot.")
        return {}

    n_rows = int(np.ceil(len(numeric_features)/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*8, n_rows*6))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    feat_data_contrib = get_feature_contributions(model, dataset)
    ideal_values = {}

    for i, feature_name in enumerate(numeric_features):
        ax = axes[i]
        f_i = feat_data_contrib[feature_name]['contributions']
        x_i = feat_data_contrib[feature_name]['original_values']

        # Ideal value
        x_opt, f_opt = get_ideal_feature_value(f_i, x_i, target=target)
        ideal_values[feature_name] = x_opt

        # Density shading (dùng RAW data)
        single_feature_data = dataset.raw_X[feature_name].to_numpy()
        x_min, x_max = x_i.min(), x_i.max()
        segments = np.linspace(x_min, x_max, n_blocks+1)
        hist, _ = np.histogram(single_feature_data, bins=segments)
        hist = hist / hist.max()
        y_min, y_max = f_i.min(), f_i.max()
        y_range = y_max - y_min
        for j in range(n_blocks):
            rect = patches.Rectangle(
                (segments[j], y_min - 0.1*y_range),
                segments[j+1]-segments[j],
                y_range*1.2,
                facecolor='gray',
                alpha=hist[j]*0.5
            )
            ax.add_patch(rect)

        # NAM function line
        ax.plot(x_i, f_i, linestyle='--', color='black', linewidth=2)

        # Red line = ideal value
        ax.axvline(x=x_opt, color='red', linestyle='-', linewidth=2)
        ax.text(x_opt, y_max, f"Ideal={x_opt:.2f}", color='red', fontsize=10, rotation=90, va='bottom')

        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel(f'f({feature_name})', fontsize=12)
        ax.set_title(feature_name, fontsize=14)

    # Remove empty axes
    for k in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    plt.show()
    return ideal_values


def load_config_from_dicts(path):
    with open(path, 'r') as f:
        cfg_dicts = yaml.safe_load(f)
    
    config = defaults()
    config.update(**cfg_dicts)
    return config


if __name__ == "__main__":
    config = load_config_from_dicts( "output/BS/0/hparams.yaml")
    dataset = NAMDataset(
            config,
            data_path=config.data_path,
            features_columns=config.features_columns,
            targets_column=config.targets_column,
        )
    
    # Debug: Check preprocessing
    print("=== DEBUG INFO ===")
    x = dataset.features
    for i in range(x.shape[1]):
        print(f"Feature {i} min: {x[:,i].min()}, max: {x[:,i].max()}, mean: {x[:,i].mean()}")
    y = dataset.targets
    print(f"Target min: {y.min()}, max: {y.max()}, mean: {y.mean()}")
    print(f"Target min: {dataset.raw_y.min()}")
    print(f"Target max: {dataset.raw_y.max()}")
    print(f"Target mean: {dataset.raw_y.mean()}")
    # Load model
    print("\n=== LOADING MODEL ===")
    model = NAM(
            config=config,
            name=config.experiment_name,
            num_inputs=len(dataset[0][0]),
            num_units=get_num_units(config, dataset.features),
        )
    litmodel = LitNAM(config, model)
    litmodel.load_state_dict(torch.load("output/BS/0/checkpoints/epoch=499-val_loss=0.7530.ckpt")['state_dict'])
    
    # Plot single feature
    # print("\n=== PLOTTING ===")
    # plot_nam_feature(litmodel.model, dataset, feature_name=config.features_columns[0], target='maximize')
    
    # Get all ideal values
    ideal_vals = get_ideal_values(litmodel.model, dataset, target='maximize')
    print("\n=== IDEAL VALUES ===")
    for feat, val in ideal_vals.items():
        print(f"{feat}: {val:.2f}")