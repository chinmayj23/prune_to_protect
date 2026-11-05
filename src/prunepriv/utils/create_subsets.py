# subset_utils.py
import os

import numpy as np
import torch
from tqdm import tqdm

def generate_subsets(
    num_subsets: int,
    total_samples: int,
    subset_percentage: float,
    base_seed: int,
    save_path: str = "subsets.pt"
):
    """
    Generates N random subsets of indices and saves them to a file.
    
    Args:
        num_subsets (int): The number of subsets (N) to generate.
        total_samples (int): Total number of samples in the full dataset (e.g., 50000 for CIFAR-10).
        subset_percentage (float): The percentage of total samples in each subset (p).
        base_seed (int): A base seed for reproducibility.
        save_path (str): The file path to save the generated subsets.
    """
    if os.path.exists(save_path):
        print(f"Subset file already exists at {save_path}. Skipping generation.")
        return

    print(f"Generating {num_subsets} subsets of {subset_percentage*100}% of the data...")
    num_samples_in_subset = int(total_samples * subset_percentage)
    all_subsets = {}

    for i in tqdm(range(num_subsets), desc="Generating Subsets"):
        # Use a different seed for each subset for variety
        subset_seed = base_seed + i
        generator = np.random.default_rng(subset_seed)
        
        # Generate random indices without replacement
        indices = generator.choice(total_samples, size=num_samples_in_subset, replace=False)
        all_subsets[i] = torch.from_numpy(indices).long() # Store as tensor
    
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print(f"Saving subsets to {save_path}...")
    torch.save(all_subsets, save_path)
    print("Done.")


def generate_complement_paired_subsets(
    num_subsets: int,
    total_samples: int,
    base_seed: int,
    save_path: str = "subsets.pt",
):
    """
    Generates complement-paired half subsets.

    This is the subset design RMIA expects when we want each training
    sample to be IN for exactly half of the reference models and OUT for
    the other half.
    """
    if os.path.exists(save_path):
        print(f"Subset file already exists at {save_path}. Skipping generation.")
        return

    if num_subsets % 2 != 0:
        raise ValueError("Complement-paired subsets require an even num_subsets.")

    print(
        f"Generating {num_subsets} complement-paired subsets with exact 50/50 "
        f"IN/OUT coverage per sample..."
    )
    num_samples_in_subset = total_samples // 2
    half_num_subsets = num_subsets // 2
    full_index_set = np.arange(total_samples)
    all_subsets = {}

    for pair_idx in tqdm(range(half_num_subsets), desc="Generating Complement Pairs"):
        subset_seed = base_seed + pair_idx
        generator = np.random.default_rng(subset_seed)
        indices = np.sort(
            generator.choice(total_samples, size=num_samples_in_subset, replace=False)
        )
        complement = np.setdiff1d(full_index_set, indices, assume_unique=False)
        all_subsets[2 * pair_idx] = torch.from_numpy(indices).long()
        all_subsets[(2 * pair_idx) + 1] = torch.from_numpy(complement).long()

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print(f"Saving subsets to {save_path}...")
    torch.save(all_subsets, save_path)
    print("Done.")

def generate_partial_subsets(
        num_subsets: int,
        total_samples: int,
        samples_for_subset: int,
        subset_percentage: int,
        base_seed: int,
        save_path: str = 'partial_subsets.pt'
):
    """
    Generates N random subsets of M fixed indices (randomly chosen once), and saves them to a file.
    For e.g, If we have N = 50000, and M = 5000, we choose M fixed indices, use them for subsetting, and keep N-M indices as it is. 
    
    Args:
        num_subsets (int): The number of subsets (N) to generate.
        total_samples (int): Total number of samples in the full dataset (e.g., 50000 for CIFAR-10).
        samples_for_subset (int): The number of samples used for subsetting
        subset_percentage (float): The percentage of total samples in each subset (p).
        base_seed (int): A base seed for reproducibility.
        save_path (str): The file path to save the generated subsets.

    Output:
        subsets as a dictionary, indices used for subseeting
    """
    if os.path.exists(save_path):
        print(f"Subset file already exists at {save_path}. Skipping generation.")
        return

    print(f"Selecting {samples_for_subset} indices for subsetting")
    generator_for_fixed_indices = np.random.default_rng(base_seed)
    subset_indices = generator_for_fixed_indices.choice(total_samples, size=samples_for_subset, replace=False)
    fixed_indices = np.setdiff1d(np.arange(total_samples), subset_indices)

    print(f"Generating {num_subsets} subsets of {subset_percentage*100}% of the data...")
    num_samples_in_subset = int(samples_for_subset * subset_percentage)
    all_subsets = {}

    for i in tqdm(range(num_subsets), desc="Generating Partial Subsets"):
        subset_seed  = base_seed+i
        generator = np.random.default_rng(subset_seed)
        indices = generator.choice(subset_indices, size = num_samples_in_subset, replace=False)
        combined_indices =  np.sort(np.concatenate([indices,fixed_indices]))
        all_subsets[i] = torch.from_numpy(combined_indices).long()

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print(f"Saving subsets to {save_path}...")
    torch.save(all_subsets, save_path)
    print("Done.")

    #I need to assert that the fixed indices are always in the subset
    fixed_tensor = torch.from_numpy(np.sort(fixed_indices)).long()

    for i, combined in all_subsets.items():
        # Extract the fixed portion by removing the subset_indices
        subset_mask = torch.isin(combined, fixed_tensor)
        extracted_fixed = combined[subset_mask]

        # Assert that the extracted fixed indices match the original
        assert torch.equal(extracted_fixed.sort().values, fixed_tensor), f"Fixed indices mismatch in subset {i}"

    return fixed_tensor
