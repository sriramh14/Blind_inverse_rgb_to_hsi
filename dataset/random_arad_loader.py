"""Fixed random subset loader compatible with the reference repository API."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset

from .dataset_loader import ARADDataset


class _SubsetWithMetadata(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], metadata: List[Dict]) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.selected_samples = metadata

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def load_random_arad1k_samples(
    root_dir: str = "data",
    num_samples: int = 50,
    seed: int = 42,
    total_images: int = 950,
    cube_key: str = "cube",
    download: bool = False,
    train_images: int = 900,
    spectral_dir: Optional[str] = None,
    rgb_dir: Optional[str] = None,
    hsi_scale: float = 1.0,
    preferred_keys: Optional[Sequence[str]] = None,
):
    dataset = ARADDataset(
        root_dir=root_dir,
        train=False,
        train_images=train_images,
        total_images=total_images,
        cube_key=cube_key,
        download=download,
        spectral_dir=spectral_dir,
        rgb_dir=rgb_dir,
        patch_size=0,
        samples_per_image=1,
        augment=False,
        hsi_scale=hsi_scale,
        preferred_keys=preferred_keys,
    )
    if num_samples > len(dataset):
        raise ValueError(f"Requested {num_samples} samples, but validation contains {len(dataset)}")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(dataset)), num_samples))
    metadata = [
        {
            "subset_index": subset_index,
            "dataset_index": dataset_index,
            "hsi_filename": Path(dataset.pairs[dataset_index][0]).name,
            "rgb_filename": Path(dataset.pairs[dataset_index][1]).name,
        }
        for subset_index, dataset_index in enumerate(indices)
    ]
    subset = _SubsetWithMetadata(dataset, indices, metadata)
    print(f"Selected {len(subset)} validation pairs using seed {seed}.")
    return subset, metadata
