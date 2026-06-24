"""Paired NTIRE/ARAD RGB-HSI loader with the same public style as the reference repo."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

HSI_EXTENSIONS = {".mat", ".h5", ".hdf5", ".npy", ".npz"}
RGB_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _canonical_stem(path: Path) -> str:
    stem = path.stem.lower()
    suffixes = (
        "_realworld", "_real_world", "_clean", "_rgb", "_spectral",
        "_hsi", "_cube", "_mat", "_image",
    )
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    return re.sub(r"[^a-z0-9]+", "", stem)


def _list_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    ext_set = {item.lower() for item in extensions}
    files = sorted(path for path in root.rglob("*") if path.suffix.lower() in ext_set)
    if not files:
        raise RuntimeError(f"No supported files were found under {root}")
    return files


def _discover_data_dir(root: Path, kind: str) -> Path:
    """Locate an RGB or spectral directory without assuming one exact release name."""
    if kind not in {"rgb", "hsi"}:
        raise ValueError("kind must be 'rgb' or 'hsi'")
    extensions = RGB_EXTENSIONS if kind == "rgb" else HSI_EXTENSIONS
    keywords = ("rgb", "realworld", "real_world", "clean") if kind == "rgb" else (
        "spectral", "hsi", "hyperspectral", "cube"
    )

    candidates: List[Tuple[int, int, Path]] = []
    for directory in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        try:
            count = sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions)
        except OSError:
            continue
        if count == 0:
            continue
        name = directory.name.lower()
        score = sum(keyword in name for keyword in keywords)
        candidates.append((score, count, directory))

    if not candidates:
        raise RuntimeError(
            f"Could not discover a {kind.upper()} directory under {root}. "
            "Set SPECTRAL_DIR and RGB_DIR explicitly in train.py."
        )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def pair_rgb_hsi(rgb_dir: Path, spectral_dir: Path, total_images: Optional[int] = None) -> List[Tuple[Path, Path]]:
    rgb_files = _list_files(rgb_dir, RGB_EXTENSIONS)
    hsi_files = _list_files(spectral_dir, HSI_EXTENSIONS)

    rgb_lookup: Dict[str, Path] = {}
    for path in rgb_files:
        key = _canonical_stem(path)
        if key in rgb_lookup:
            raise RuntimeError(f"Duplicate RGB scene id '{key}': {rgb_lookup[key]} and {path}")
        rgb_lookup[key] = path

    hsi_lookup: Dict[str, Path] = {}
    for path in hsi_files:
        key = _canonical_stem(path)
        if key in hsi_lookup:
            raise RuntimeError(f"Duplicate HSI scene id '{key}': {hsi_lookup[key]} and {path}")
        hsi_lookup[key] = path

    common = sorted(set(rgb_lookup) & set(hsi_lookup))
    if not common:
        raise RuntimeError(
            "No paired RGB-HSI scenes were found. "
            f"Example RGB names: {[p.name for p in rgb_files[:5]]}; "
            f"example HSI names: {[p.name for p in hsi_files[:5]]}"
        )
    if total_images is not None:
        common = common[: int(total_images)]
    return [(hsi_lookup[key], rgb_lookup[key]) for key in common]


def _select_3d_array(mapping: Dict[str, np.ndarray], preferred_keys: Sequence[str], bands: int) -> np.ndarray:
    for key in preferred_keys:
        value = mapping.get(key)
        if isinstance(value, np.ndarray) and value.ndim == 3:
            return value
    candidates = [
        value for key, value in mapping.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 3
    ]
    if not candidates:
        raise KeyError("No 3-D array was found in the HSI file")
    candidates.sort(key=lambda value: (bands not in value.shape, -value.size))
    return candidates[0]


def _load_hdf5(path: Path, preferred_keys: Sequence[str], bands: int) -> np.ndarray:
    arrays: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        def visitor(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                arrays[name.split("/")[-1]] = np.asarray(obj)
        handle.visititems(visitor)
    return _select_3d_array(arrays, preferred_keys, bands)


def _load_hsi(path: Path, preferred_keys: Sequence[str], bands: int, scale: float) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        array = _select_3d_array({key: archive[key] for key in archive.files}, preferred_keys, bands)
    elif suffix in {".h5", ".hdf5"}:
        array = _load_hdf5(path, preferred_keys, bands)
    elif suffix == ".mat":
        try:
            array = _select_3d_array(loadmat(path), preferred_keys, bands)
        except (NotImplementedError, ValueError, OSError):
            array = _load_hdf5(path, preferred_keys, bands)
    else:
        raise ValueError(f"Unsupported HSI extension: {suffix}")

    array = np.asarray(array, dtype=np.float32)
    if bands not in array.shape:
        raise ValueError(f"Expected one axis of {path.name} to have {bands} bands, got {array.shape}")
    band_axis = list(array.shape).index(bands)
    array = np.moveaxis(array, band_axis, 0)
    if scale <= 0:
        raise ValueError("hsi_scale must be positive")
    return np.ascontiguousarray(array / float(scale))


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def _align(rgb: np.ndarray, hsi: np.ndarray, rgb_path: Path, hsi_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if rgb.shape[1:] == hsi.shape[1:]:
        return rgb, hsi
    if rgb.shape[1:] == hsi.shape[1:][::-1]:
        return rgb, np.ascontiguousarray(hsi.transpose(0, 2, 1))
    raise ValueError(
        f"Spatial mismatch between {rgb_path.name} {rgb.shape} and {hsi_path.name} {hsi.shape}"
    )


def _paired_crop(rgb: torch.Tensor, hsi: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = rgb.shape[-2:]
    if patch_size <= 0:
        return rgb, hsi
    if height < patch_size or width < patch_size:
        raise ValueError(f"Patch size {patch_size} exceeds image size {(height, width)}")
    top = random.randint(0, height - patch_size)
    left = random.randint(0, width - patch_size)
    return (
        rgb[:, top : top + patch_size, left : left + patch_size],
        hsi[:, top : top + patch_size, left : left + patch_size],
    )


def _paired_augment(rgb: torch.Tensor, hsi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        rgb, hsi = torch.flip(rgb, (-1,)), torch.flip(hsi, (-1,))
    if random.random() < 0.5:
        rgb, hsi = torch.flip(rgb, (-2,)), torch.flip(hsi, (-2,))
    rotations = random.randint(0, 3)
    if rotations:
        rgb = torch.rot90(rgb, rotations, (-2, -1))
        hsi = torch.rot90(hsi, rotations, (-2, -1))
    return rgb.contiguous(), hsi.contiguous()


class ARADDataset(Dataset):
    """NTIRE/ARAD paired dataset retaining the reference repository's class name.

    The loader can split one combined directory by index, or use explicit
    ``spectral_dir``/``rgb_dir`` paths. Training samples use aligned random
    crops; validation samples retain full resolution.
    """

    def __init__(
        self,
        root_dir: str = "data",
        train: bool = True,
        train_images: int = 900,
        total_images: int = 950,
        cube_key: str = "cube",
        download: bool = False,
        spectral_dir: Optional[str] = None,
        rgb_dir: Optional[str] = None,
        patch_size: int = 128,
        samples_per_image: int = 8,
        augment: bool = True,
        hsi_scale: float = 1.0,
        clamp_hsi: bool = True,
        num_bands: int = 31,
        preferred_keys: Optional[Sequence[str]] = None,
    ) -> None:
        del download  # NTIRE 2022 is not downloaded automatically by this repository.
        root = Path(root_dir)
        self.spectral_dir = Path(spectral_dir) if spectral_dir else _discover_data_dir(root, "hsi")
        self.rgb_dir = Path(rgb_dir) if rgb_dir else _discover_data_dir(root, "rgb")
        self.preferred_keys = tuple(preferred_keys or (cube_key, "rad", "hsi", "spectral", "data"))
        self.num_bands = int(num_bands)
        self.patch_size = int(patch_size)
        self.samples_per_image = int(samples_per_image) if train else 1
        self.training = bool(train)
        self.augment = bool(augment)
        self.hsi_scale = float(hsi_scale)
        self.clamp_hsi = bool(clamp_hsi)

        pairs = pair_rgb_hsi(self.rgb_dir, self.spectral_dir, total_images)
        if train_images < 1 or train_images >= len(pairs):
            raise ValueError(
                f"train_images must be in [1,{len(pairs)-1}] for a combined split; got {train_images}. "
                "Adjust TRAIN_IMAGES/TOTAL_IMAGES in train.py."
            )
        self.pairs = pairs[:train_images] if train else pairs[train_images:]
        split_name = "Train" if train else "Validation"
        print(
            f"{split_name}: {len(self.pairs)} scenes | RGB={self.rgb_dir} | HSI={self.spectral_dir}"
        )

    def __len__(self) -> int:
        return len(self.pairs) * self.samples_per_image

    def __getitem__(self, index: int):
        pair_index = index // self.samples_per_image
        hsi_path, rgb_path = self.pairs[pair_index]
        hsi_np = _load_hsi(
            hsi_path, self.preferred_keys, self.num_bands, self.hsi_scale
        )
        rgb_np = _load_rgb(rgb_path)
        rgb_np, hsi_np = _align(rgb_np, hsi_np, rgb_path, hsi_path)
        rgb = torch.from_numpy(rgb_np).float()
        hsi = torch.from_numpy(hsi_np).float()
        if self.clamp_hsi:
            hsi = hsi.clamp(0.0, 1.0)
        if self.training:
            rgb, hsi = _paired_crop(rgb, hsi, self.patch_size)
            if self.augment:
                rgb, hsi = _paired_augment(rgb, hsi)
        return rgb, hsi
