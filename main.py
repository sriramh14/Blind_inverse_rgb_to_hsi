#!/usr/bin/env python3
"""Train, validate, or test residual latent diffusion for NTIRE RGB-to-HSI.

Only two command-line arguments are exposed:

    python main.py --mode train --stage 1
    python main.py --mode train --stage 2
    python main.py --mode train --stage 3
    python main.py --mode val   --stage 3
    python main.py --mode test  --stage 3

Stages
------
1. Train a high-fidelity 31-band HSI autoencoder.
2. Freeze Stage 1 and train the RGB latent initializer together with the
   differentiable HSI-to-RGB forward adapter.
3. Freeze Stages 1-2 and train conditional diffusion on the missing latent
   residual. Validation/test use deterministic DDIM and optional decoded
   RGB-consistency guidance.

The CONFIG section controls data paths, splitting, optimizers, architecture,
and loss weights. ``DATA_BACKEND='inline'`` uses the robust full-resolution
NTIRE loader in this file. ``DATA_BACKEND='repo'`` keeps compatibility with the
``dataset/`` loaders from the uploaded reference repository.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import torch
import torch.nn.functional as F
from scipy.io import savemat
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from dataset.dataset_loader import ARADDataset
from dataset.random_arad_loader import load_random_arad1k_samples
from loss import compute_metrics, l1_loss, mrae, mse_loss, sam, ssim
from models import LatentDiffusionRGB2HSI, ModelConfig


# ==================================================
# CONFIG
# ==================================================

MODE = "train"                 # "train", "val", or "test"; "eval" aliases "test"
STAGE = 1                       # 1: autoencoder, 2: RGB initializer, 3: latent diffusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
VAL_SEED = 1234
SPLIT_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
# "inline": robust paired loader and persistent 80/10/10 scene split below.
# "repo":   ARADDataset and random_arad_loader retained from the uploaded repo.
DATA_BACKEND = "inline"

DATA_ROOT: Union[str, Path] = os.environ.get(
    "ARAD_DATA_ROOT",
    "/kaggle/input/datasets/sriramhari14/ntire-2022",
)
HSI_KEY = "cube"
NUM_BANDS = 31
TOTAL_IMAGES: Optional[int] = None

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SPLIT_DIR = PROJECT_ROOT / "splits"
SPLIT_FILE = SPLIT_DIR / "arad_train_val_test_split.pth"
VALIDATE_DATASET_FILES = True
SKIP_INVALID_PAIRS = True
DATASET_VALIDATION_CACHE = SPLIT_DIR / "ntire_valid_pairs_cache.pth"
DATASET_VALIDATION_PROGRESS = 100

# Full-resolution by default. Set an integer (for example 128) to use paired
# random crops while retaining this same training script.
TRAIN_PATCH_SIZE: Optional[int] = None
USE_GEOMETRIC_AUGMENTATION = True
PAD_MULTIPLE = 16
BATCH_SIZE = 2
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
NUM_WORKERS = 2
PREFETCH_FACTOR = 2
PIN_MEMORY = DEVICE == "cuda"
PERSISTENT_WORKERS = True
PROGRESS_EVERY_N_BATCHES = 30
TRAIN_METRICS_EVERY_N_BATCHES = PROGRESS_EVERY_N_BATCHES

# Compatibility settings for DATA_BACKEND="repo".
REPO_TRAIN_IMAGES = 900
REPO_TOTAL_IMAGES = 950
REPO_SAMPLES_PER_IMAGE = 1
REPO_SPECTRAL_DIR: Optional[str] = None
REPO_RGB_DIR: Optional[str] = None
REPO_HSI_SCALE = 1.0
REPO_DOWNLOAD_DATA = False
REPO_SPLIT_HELDOUT_FOR_TEST = True
REPO_TEST_USE_RANDOM_ARAD_LOADER = False
REPO_RANDOM_TEST_IMAGES = 25
REPO_RANDOM_TEST_TOTAL_IMAGES = 950

HSI_VALUE_SCALE: Optional[float] = None
CLIP_INPUTS_TO_UNIT_RANGE = True
MRAE_EPS = 1e-6
SSIM_WINDOW_SIZE = 3

# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------
STAGE1_EPOCHS = 100
STAGE2_EPOCHS = 100
STAGE3_EPOCHS = 100
STAGE1_LR = 2e-4
STAGE2_LR = 2e-4
STAGE3_LR = 1e-4
WEIGHT_DECAY = 0.0
MIN_LR = 1e-7
GRAD_CLIP_NORM = 1.0
USE_AMP = True
USE_GRADIENT_CHECKPOINTING = False
EARLY_STOPPING_PATIENCE = 20
LATENT_STATS_MAX_BATCHES: Optional[int] = 200

# Stage 1 autoencoder objective.
AE_L1_WEIGHT = 1.0
AE_MRAE_WEIGHT = 0.20
AE_SAM_WEIGHT = 0.01/90
AE_SPECTRAL_GRAD_WEIGHT = 0.10

# Stage 2 RGB latent initializer and forward adapter objective.
INIT_LATENT_L1_WEIGHT = 1.0
INIT_HSI_MRAE_WEIGHT = 1.0
INIT_HSI_L1_WEIGHT = 0.10
INIT_HSI_SAM_WEIGHT = 0.01
FORWARD_RGB_L1_WEIGHT = 1.0
FORWARD_RGB_MSE_WEIGHT = 0.25
FORWARD_RGB_SSIM_WEIGHT = 0.10
FORWARD_SMOOTHNESS_WEIGHT = 1e-3

# Stage 3 residual diffusion objective.
LAMBDA_NOISE = 1.0
LAMBDA_RESIDUAL_X0 = 0.10
LAMBDA_DECODED_MRAE = 0.05
DECODED_LOSS_MAX_T_FRACTION = 0.50
USE_EMA = True
EMA_DECAY = 0.999

# Stage 3 validation can be expensive. None evaluates the complete split.
DIFFUSION_VAL_MAX_IMAGES: Optional[int] = 5
DIFFUSION_VAL_SAMPLING_STEPS = 10
EVAL_MAX_IMAGES: Optional[int] = None
SAVE_TEST_PREDICTIONS = True
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# -----------------------------------------------------------------------------
# ARCHITECTURE
# -----------------------------------------------------------------------------
AE_BASE_CHANNELS = 64
LATENT_CHANNELS = 16
RGB_BASE_CHANNELS = 48
DIFFUSION_BASE_CHANNELS = 96
DIFFUSION_CHANNEL_MULTS = (1, 2, 4)
TIME_EMBEDDING_DIM = 384
ATTENTION_HEADS = 4
DROPOUT = 0.0

DIFFUSION_TIMESTEPS = 1000
BETA_SCHEDULE = "cosine"
LINEAR_BETA_START = 1e-4
LINEAR_BETA_END = 2e-2
SAMPLING_STEPS = 25
DDIM_ETA = 0.0
PHYSICS_GUIDANCE_SCALE = 0.03
NORMALIZE_GUIDANCE = True
PHYSICS_BLUR_KERNEL = 5
CLIP_DENOISED = True
HSI_MIN = 0.0
HSI_MAX = 1.0

# -----------------------------------------------------------------------------
# CHECKPOINTS
# -----------------------------------------------------------------------------
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
STAGE1_BEST_PATH = CHECKPOINT_DIR / "latent_stage1_autoencoder_best.pth"
STAGE1_BEST_LOSS_PATH = CHECKPOINT_DIR / "latent_stage1_autoencoder_best_loss.pth"
STAGE1_LATEST_PATH = CHECKPOINT_DIR / "latent_stage1_autoencoder_latest.pth"
STAGE2_BEST_PATH = CHECKPOINT_DIR / "latent_stage2_initializer_best.pth"
STAGE2_BEST_LOSS_PATH = CHECKPOINT_DIR / "latent_stage2_initializer_best_loss.pth"
STAGE2_LATEST_PATH = CHECKPOINT_DIR / "latent_stage2_initializer_latest.pth"
STAGE3_BEST_PATH = CHECKPOINT_DIR / "latent_stage3_diffusion_best.pth"
STAGE3_BEST_LOSS_PATH = CHECKPOINT_DIR / "latent_stage3_diffusion_best_loss.pth"
STAGE3_LATEST_PATH = CHECKPOINT_DIR / "latent_stage3_diffusion_latest.pth"

STAGE1_TEACHER_CHECKPOINT: Union[str, Path] = STAGE1_BEST_PATH
STAGE2_TEACHER_CHECKPOINT: Union[str, Path] = STAGE2_BEST_PATH
RESUME_CHECKPOINT: Optional[Union[str, Path]] = None
EVAL_CHECKPOINT: Optional[Union[str, Path]] = None

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# COMMAND LINE
# ==================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual latent diffusion RGB-to-HSI")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        default=STAGE,
        help="1=HSI autoencoder, 2=RGB latent initializer, 3=residual latent diffusion.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "val", "test", "eval"],
        default=MODE,
        help="Run mode. eval is retained as an alias for test.",
    )
    return parser.parse_args()


# ==================================================
# REPRODUCIBILITY AND SMALL UTILITIES
# ==================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_model_config() -> ModelConfig:
    return ModelConfig(
        num_bands=NUM_BANDS,
        hsi_min=HSI_MIN,
        hsi_max=HSI_MAX,
        ae_base_channels=AE_BASE_CHANNELS,
        latent_channels=LATENT_CHANNELS,
        rgb_base_channels=RGB_BASE_CHANNELS,
        diffusion_base_channels=DIFFUSION_BASE_CHANNELS,
        diffusion_channel_mults=DIFFUSION_CHANNEL_MULTS,
        time_embedding_dim=TIME_EMBEDDING_DIM,
        attention_heads=ATTENTION_HEADS,
        dropout=DROPOUT,
        diffusion_timesteps=DIFFUSION_TIMESTEPS,
        beta_schedule=BETA_SCHEDULE,
        linear_beta_start=LINEAR_BETA_START,
        linear_beta_end=LINEAR_BETA_END,
        sampling_steps=SAMPLING_STEPS,
        ddim_eta=DDIM_ETA,
        physics_guidance_scale=PHYSICS_GUIDANCE_SCALE,
        normalize_guidance=NORMALIZE_GUIDANCE,
        physics_blur_kernel=PHYSICS_BLUR_KERNEL,
        clip_denoised=CLIP_DENOISED,
    )


def make_grad_scaler(enabled: bool):
    active = enabled and DEVICE == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=active)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=active)


@contextmanager
def autocast_context(enabled: bool):
    if DEVICE == "cuda":
        try:
            with torch.amp.autocast("cuda", enabled=enabled):
                yield
        except (AttributeError, TypeError):
            with torch.cuda.amp.autocast(enabled=enabled):
                yield
    else:
        yield


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def count_parameters(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def make_orig_hw_tensor(orig_hw: Optional[Any], hsi: torch.Tensor) -> torch.Tensor:
    batch_size = hsi.shape[0]
    if orig_hw is None:
        return torch.tensor([[hsi.shape[-2], hsi.shape[-1]]] * batch_size, dtype=torch.long)
    value = orig_hw.detach().cpu() if torch.is_tensor(orig_hw) else torch.as_tensor(orig_hw)
    if value.ndim == 1:
        value = value.view(1, 2).repeat(batch_size, 1)
    if value.shape[0] == 1 and batch_size > 1:
        value = value.repeat(batch_size, 1)
    if value.ndim != 2 or value.shape != (batch_size, 2):
        raise ValueError(f"orig_hw must have shape [B,2], got {tuple(value.shape)}")
    return value.long()


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Any, Optional[torch.Tensor]]:
    name = None
    orig_hw = None
    if isinstance(batch, dict):
        rgb = batch.get("rgb", batch.get("lq"))
        hsi = batch.get("hsi", batch.get("gt"))
        name = batch.get("name", batch.get("filename"))
        orig_hw = batch.get("orig_hw", batch.get("original_hw"))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        rgb, hsi = batch[0], batch[1]
        if len(batch) >= 3:
            name = batch[2]
        if len(batch) >= 4:
            orig_hw = batch[3]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch).__name__}")
    if not torch.is_tensor(rgb) or not torch.is_tensor(hsi):
        raise TypeError("RGB and HSI must be tensors after DataLoader collation")
    if rgb.ndim != 4 or hsi.ndim != 4:
        raise ValueError(f"Expected RGB/HSI batches, got {tuple(rgb.shape)}, {tuple(hsi.shape)}")
    if rgb.shape[1] != 3 or hsi.shape[1] != NUM_BANDS:
        raise ValueError(f"Expected channels 3/{NUM_BANDS}, got {rgb.shape[1]}/{hsi.shape[1]}")
    return rgb, hsi, name, orig_hw


def crop_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    orig_hw: torch.Tensor,
    sample_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    height = int(orig_hw[sample_index, 0])
    width = int(orig_hw[sample_index, 1])
    return (
        pred[sample_index : sample_index + 1, :, :height, :width],
        target[sample_index : sample_index + 1, :, :height, :width],
    )


TRAIN_DISPLAY_METRICS = ("mrae", "rmse", "psnr", "sam", "ssim")


@torch.no_grad()
def compute_batch_display_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    orig_hw: torch.Tensor,
) -> Dict[str, float]:
    totals = {name: 0.0 for name in TRAIN_DISPLAY_METRICS}
    for sample_index in range(pred.shape[0]):
        sample_pred, sample_target = crop_sample(pred, target, orig_hw, sample_index)
        metrics = compute_metrics(
            sample_pred.float(),
            sample_target.float(),
            mrae_eps=MRAE_EPS,
            ssim_window_size=SSIM_WINDOW_SIZE,
        )
        for name in TRAIN_DISPLAY_METRICS:
            totals[name] += metrics[name]
    return {name: value / max(pred.shape[0], 1) for name, value in totals.items()}


# ==================================================
# INLINE FULL-RESOLUTION NTIRE DATA LOADER
# ==================================================

RGB_DIRECTORY_NAMES = ("Train_RGB", "train_rgb", "RGB", "rgb")
HSI_DIRECTORY_NAMES = (
    "Train_spectral", "Train_Spectral", "train_spectral",
    "Train_HSI", "train_hsi", "HSI", "hsi",
)
RGB_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HSI_SUFFIXES = {".mat", ".h5", ".hdf5", ".npy", ".npz"}


def resolve_data_root() -> Path:
    root = Path(DATA_ROOT).expanduser()
    root = (PROJECT_ROOT / root).resolve() if not root.is_absolute() else root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")
    return root


def _find_named_directory(root: Path, names: Sequence[str], kind: str) -> Path:
    wanted = {name.lower() for name in names}
    direct = [path for path in root.iterdir() if path.is_dir()]
    for path in direct:
        if path.name.lower() in wanted:
            return path
    for wrapper in direct:
        try:
            children = [path for path in wrapper.iterdir() if path.is_dir()]
        except PermissionError:
            continue
        for path in children:
            if path.name.lower() in wanted:
                return path
    raise FileNotFoundError(
        f"Could not find {kind} directory under {root}; expected one of {list(names)}"
    )


def _collect_files(directory: Path, suffixes: set[str]) -> list[Path]:
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
    if not files:
        raise FileNotFoundError(f"No supported files found under {directory}")
    return files


def _normalized_pair_key(path: Path) -> str:
    key = path.stem.lower().strip()
    removable = (
        "_spectral", "-spectral", "_spectrum", "-spectrum",
        "_realworld", "-realworld", "_clean", "-clean",
        "_rgb", "-rgb", "_hsi", "-hsi", "_hyperspectral",
    )
    changed = True
    while changed:
        changed = False
        for suffix in removable:
            if key.endswith(suffix):
                key = key[: -len(suffix)]
                changed = True
                break
    return key


def _build_unique_map(files: list[Path], kind: str) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for path in files:
        key = _normalized_pair_key(path)
        if key in result:
            raise RuntimeError(f"Duplicate normalized {kind} key '{key}': {result[key]} and {path}")
        result[key] = path
    return result


def discover_paired_samples(root: Path) -> list[Tuple[Path, Path, str]]:
    rgb_dir = _find_named_directory(root, RGB_DIRECTORY_NAMES, "RGB")
    hsi_dir = _find_named_directory(root, HSI_DIRECTORY_NAMES, "HSI")
    rgb_map = _build_unique_map(_collect_files(rgb_dir, RGB_SUFFIXES), "RGB")
    hsi_map = _build_unique_map(_collect_files(hsi_dir, HSI_SUFFIXES), "HSI")
    common = sorted(set(rgb_map) & set(hsi_map))
    if not common:
        raise RuntimeError("No paired RGB/HSI samples could be matched by filename stem")
    pairs = [(rgb_map[key], hsi_map[key], key) for key in common]
    if TOTAL_IMAGES is not None:
        pairs = pairs[: int(TOTAL_IMAGES)]
    print(f"Dataset root: {root}")
    print(f"RGB directory: {rgb_dir}")
    print(f"HSI directory: {hsi_dir}")
    print(f"Matched RGB-HSI pairs: {len(pairs)}")
    return pairs


def _pair_fingerprint(pairs: list[Tuple[Path, Path, str]]) -> str:
    records = []
    for rgb, hsi, name in pairs:
        rs, hs = rgb.stat(), hsi.stat()
        records.append(
            f"{name}|{rgb.resolve()}|{rs.st_size}|{rs.st_mtime_ns}|"
            f"{hsi.resolve()}|{hs.st_size}|{hs.st_mtime_ns}"
        )
    return hashlib.sha256("\n".join(records).encode()).hexdigest()


def _select_3d_array(mapping: Dict[str, Any], cube_key: str, path: Path) -> np.ndarray:
    if cube_key in mapping and isinstance(mapping[cube_key], np.ndarray) and mapping[cube_key].ndim == 3:
        return mapping[cube_key]
    candidates = [
        value for key, value in mapping.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 3
    ]
    if not candidates:
        raise KeyError(f"No 3-D numeric cube found in {path}")
    candidates.sort(key=lambda value: (NUM_BANDS not in value.shape, -value.size))
    return candidates[0]


def _load_hdf5_cube(path: Path, cube_key: str) -> np.ndarray:
    import h5py
    arrays: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        if cube_key in handle and isinstance(handle[cube_key], h5py.Dataset):
            return np.asarray(handle[cube_key])
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                arrays[name] = np.asarray(obj)
        handle.visititems(visitor)
    return _select_3d_array(arrays, cube_key, path)


def load_hsi_cube(path: Path, cube_key: str, rgb_hw: Tuple[int, int]) -> np.ndarray:
    suffix = path.suffix.lower()
    original_dtype = None
    if suffix == ".npy":
        raw = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        raw = _select_3d_array({key: archive[key] for key in archive.files}, cube_key, path)
    elif suffix in {".h5", ".hdf5"}:
        raw = _load_hdf5_cube(path, cube_key)
    elif suffix == ".mat":
        try:
            from scipy.io import loadmat
            raw = _select_3d_array(loadmat(path), cube_key, path)
        except (NotImplementedError, ValueError, OSError):
            raw = _load_hdf5_cube(path, cube_key)
    else:
        raise ValueError(f"Unsupported HSI file: {path}")

    original_dtype = np.asarray(raw).dtype
    raw = np.asarray(raw)
    if raw.ndim != 3 or NUM_BANDS not in raw.shape:
        raise ValueError(f"Expected a 3-D {NUM_BANDS}-band cube in {path}, got {raw.shape}")

    rgb_h, rgb_w = rgb_hw
    aligned = None
    for axis, size in enumerate(raw.shape):
        if size != NUM_BANDS:
            continue
        chw = np.moveaxis(raw, axis, 0)
        if chw.shape[1:] == (rgb_h, rgb_w):
            aligned = chw
            break
        if chw.shape[1:] == (rgb_w, rgb_h):
            aligned = chw.transpose(0, 2, 1)
            break
    if aligned is None:
        raise ValueError(f"Spatial mismatch in {path.name}: RGB {(rgb_h, rgb_w)}, HSI {raw.shape}")

    cube = np.asarray(aligned, dtype=np.float32)
    if np.issubdtype(original_dtype, np.integer):
        cube /= float(np.iinfo(original_dtype).max)
    elif HSI_VALUE_SCALE is not None:
        cube /= float(HSI_VALUE_SCALE)
    if not np.isfinite(cube).all():
        cube = np.nan_to_num(cube, nan=0.0, posinf=1.0, neginf=0.0)
    if CLIP_INPUTS_TO_UNIT_RANGE:
        cube = np.clip(cube, HSI_MIN, HSI_MAX)
    return np.ascontiguousarray(cube)


def _shape_contains_valid_cube(shape: Tuple[int, ...], rgb_hw: Tuple[int, int]) -> bool:
    if len(shape) != 3 or NUM_BANDS not in shape:
        return False
    rgb_h, rgb_w = rgb_hw
    for spectral_axis, size in enumerate(shape):
        if size != NUM_BANDS:
            continue
        spatial = tuple(shape[index] for index in range(3) if index != spectral_axis)
        if spatial in {(rgb_h, rgb_w), (rgb_w, rgb_h)}:
            return True
    return False


def _inspect_hdf5_cube(path: Path, cube_key: str, rgb_hw: Tuple[int, int]) -> None:
    import h5py
    candidates: list[Tuple[str, Tuple[int, ...]]] = []
    with h5py.File(path, "r") as handle:
        if cube_key in handle and isinstance(handle[cube_key], h5py.Dataset):
            candidates.append((cube_key, tuple(int(value) for value in handle[cube_key].shape)))
        else:
            def visitor(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                    candidates.append((name, tuple(int(value) for value in obj.shape)))
            handle.visititems(visitor)
    if not any(_shape_contains_valid_cube(shape, rgb_hw) for _, shape in candidates):
        raise ValueError(
            f"No {NUM_BANDS}-band HDF5 cube in {path.name} matches RGB size {rgb_hw}; "
            f"datasets={candidates}"
        )


def _inspect_hsi_metadata(path: Path, cube_key: str, rgb_hw: Tuple[int, int]) -> None:
    suffix = path.suffix.lower()
    if suffix == ".mat":
        try:
            from scipy.io import whosmat
            metadata = whosmat(path)
            candidates = [
                (name, tuple(int(value) for value in shape))
                for name, shape, _ in metadata
                if len(shape) == 3
            ]
            preferred = [item for item in candidates if item[0] == cube_key]
            checked = preferred if preferred else candidates
            if not any(_shape_contains_valid_cube(shape, rgb_hw) for _, shape in checked):
                raise ValueError(
                    f"No {NUM_BANDS}-band MAT cube in {path.name} matches RGB size {rgb_hw}; "
                    f"arrays={candidates}"
                )
            return
        except (NotImplementedError, ValueError, OSError):
            _inspect_hdf5_cube(path, cube_key, rgb_hw)
            return
    if suffix in {".h5", ".hdf5"}:
        _inspect_hdf5_cube(path, cube_key, rgb_hw)
        return
    # NPY/NPZ headers are cheap to inspect and do not require loading all values.
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r")
        if not _shape_contains_valid_cube(tuple(array.shape), rgb_hw):
            raise ValueError(f"Invalid NPY cube shape {array.shape} for RGB size {rgb_hw}")
        return
    if suffix == ".npz":
        with np.load(path) as archive:
            shapes = [tuple(archive[key].shape) for key in archive.files]
        if not any(_shape_contains_valid_cube(shape, rgb_hw) for shape in shapes):
            raise ValueError(f"No valid NPZ cube in {path.name}; shapes={shapes}")
        return
    raise ValueError(f"Unsupported HSI metadata format: {path}")


def _validate_one_pair(rgb_path: Path, hsi_path: Path, cube_key: str) -> None:
    from PIL import Image
    with Image.open(rgb_path) as image:
        width, height = image.size
        image.verify()
    _inspect_hsi_metadata(hsi_path, cube_key, (height, width))


def validate_paired_samples(
    pairs: list[Tuple[Path, Path, str]],
) -> list[Tuple[Path, Path, str]]:
    if not VALIDATE_DATASET_FILES:
        return pairs
    fingerprint = _pair_fingerprint(pairs)
    pair_by_name = {name: (rgb, hsi, name) for rgb, hsi, name in pairs}
    if DATASET_VALIDATION_CACHE.exists():
        try:
            cached = torch.load(DATASET_VALIDATION_CACHE, map_location="cpu", weights_only=False)
        except TypeError:
            cached = torch.load(DATASET_VALIDATION_CACHE, map_location="cpu")
        if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
            valid = [pair_by_name[name] for name in cached.get("valid_names", []) if name in pair_by_name]
            if valid:
                print(f"Dataset integrity cache: {len(valid)} valid pairs")
                return valid

    from PIL import Image
    valid: list[Tuple[Path, Path, str]] = []
    invalid: list[Dict[str, str]] = []
    print(f"Validating {len(pairs)} paired files...", flush=True)
    for index, (rgb_path, hsi_path, name) in enumerate(pairs, 1):
        try:
            _validate_one_pair(rgb_path, hsi_path, HSI_KEY)
            valid.append((rgb_path, hsi_path, name))
        except Exception as exc:
            invalid.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"Invalid pair excluded: {name}: {invalid[-1]['error']}")
            if not SKIP_INVALID_PAIRS:
                raise
        if index % DATASET_VALIDATION_PROGRESS == 0 or index == len(pairs):
            print(f"Dataset validation {index}/{len(pairs)} | valid {len(valid)} | invalid {len(invalid)}")
    if not valid:
        raise RuntimeError("Every dataset pair failed validation")
    torch.save(
        {"fingerprint": fingerprint, "valid_names": [name for _, _, name in valid], "invalid": invalid},
        DATASET_VALIDATION_CACHE,
    )
    return valid


class NTIREPairedDataset(Dataset):
    def __init__(
        self,
        pairs: list[Tuple[Path, Path, str]],
        *,
        training: bool,
        cube_key: str,
    ) -> None:
        self.pairs = pairs
        self.training = training
        self.cube_key = cube_key

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        from PIL import Image
        rgb_path, hsi_path, name = self.pairs[index]
        with Image.open(rgb_path) as image:
            rgb_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        original_h, original_w = rgb_array.shape[:2]
        hsi_array = load_hsi_cube(hsi_path, self.cube_key, (original_h, original_w))
        rgb = torch.from_numpy(np.ascontiguousarray(rgb_array.transpose(2, 0, 1)))
        hsi = torch.from_numpy(hsi_array)

        if self.training and TRAIN_PATCH_SIZE is not None:
            patch = int(TRAIN_PATCH_SIZE)
            if patch > original_h or patch > original_w:
                raise ValueError(f"TRAIN_PATCH_SIZE={patch} exceeds {(original_h, original_w)}")
            top = random.randint(0, original_h - patch)
            left = random.randint(0, original_w - patch)
            rgb = rgb[:, top : top + patch, left : left + patch]
            hsi = hsi[:, top : top + patch, left : left + patch]

        if self.training and USE_GEOMETRIC_AUGMENTATION:
            if random.random() < 0.5:
                rgb, hsi = torch.flip(rgb, (-1,)), torch.flip(hsi, (-1,))
            if random.random() < 0.5:
                rgb, hsi = torch.flip(rgb, (-2,)), torch.flip(hsi, (-2,))
            if TRAIN_PATCH_SIZE is not None:
                rotations = random.randint(0, 3)
                if rotations:
                    rgb = torch.rot90(rgb, rotations, (-2, -1))
                    hsi = torch.rot90(hsi, rotations, (-2, -1))

        return {
            "rgb": rgb.contiguous(),
            "hsi": hsi.contiguous(),
            "name": name,
            "orig_hw": torch.tensor(rgb.shape[-2:], dtype=torch.long),
        }


def _safe_pad(tensor: torch.Tensor, pad_right: int, pad_bottom: int) -> torch.Tensor:
    if pad_right == 0 and pad_bottom == 0:
        return tensor
    mode = "reflect" if tensor.shape[-2] > pad_bottom and tensor.shape[-1] > pad_right else "replicate"
    return F.pad(tensor, (0, pad_right, 0, pad_bottom), mode=mode)


def collate_paired_batch(samples: list[Any]) -> Dict[str, Any]:
    normalized = []
    for index, sample in enumerate(samples):
        if isinstance(sample, dict):
            rgb = sample.get("rgb", sample.get("lq"))
            hsi = sample.get("hsi", sample.get("gt"))
            name = sample.get("name", f"sample_{index}")
            orig_hw = sample.get("orig_hw", torch.tensor(rgb.shape[-2:]))
        elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
            rgb, hsi = sample[0], sample[1]
            name = sample[2] if len(sample) >= 3 else f"sample_{index}"
            orig_hw = sample[3] if len(sample) >= 4 else torch.tensor(rgb.shape[-2:])
        else:
            raise TypeError(f"Unsupported sample type: {type(sample).__name__}")
        normalized.append({"rgb": rgb, "hsi": hsi, "name": name, "orig_hw": torch.as_tensor(orig_hw)})

    max_h = max(int(sample["rgb"].shape[-2]) for sample in normalized)
    max_w = max(int(sample["rgb"].shape[-1]) for sample in normalized)
    padded_h = ceil_div(max_h, PAD_MULTIPLE) * PAD_MULTIPLE
    padded_w = ceil_div(max_w, PAD_MULTIPLE) * PAD_MULTIPLE
    rgb_batch, hsi_batch, names, sizes = [], [], [], []
    for sample in normalized:
        rgb, hsi = sample["rgb"], sample["hsi"]
        if rgb.shape[-2:] != hsi.shape[-2:]:
            raise ValueError(f"RGB/HSI mismatch: {tuple(rgb.shape)} vs {tuple(hsi.shape)}")
        pad_bottom = padded_h - rgb.shape[-2]
        pad_right = padded_w - rgb.shape[-1]
        rgb_batch.append(_safe_pad(rgb, pad_right, pad_bottom))
        hsi_batch.append(_safe_pad(hsi, pad_right, pad_bottom))
        names.append(sample["name"])
        sizes.append(torch.as_tensor(sample["orig_hw"], dtype=torch.long))
    return {
        "rgb": torch.stack(rgb_batch),
        "hsi": torch.stack(hsi_batch),
        "name": names,
        "orig_hw": torch.stack(sizes),
    }


def compute_split_lengths(total: int) -> Tuple[int, int, int]:
    if not math.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0, abs_tol=1e-8):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1")
    train_count = int(round(total * TRAIN_RATIO))
    val_count = int(round(total * VAL_RATIO))
    test_count = total - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError("Configured split produces an empty subset")
    return train_count, val_count, test_count


def _dataset_fingerprint(pairs: list[Tuple[Path, Path, str]]) -> str:
    payload = "\n".join(f"{name}|{rgb.name}|{hsi.name}" for rgb, hsi, name in pairs)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_or_create_split_indices(pairs: list[Tuple[Path, Path, str]]) -> Dict[str, list[int]]:
    total = len(pairs)
    train_count, val_count, test_count = compute_split_lengths(total)
    expected = {
        "total": total,
        "seed": SPLIT_SEED,
        "counts": (train_count, val_count, test_count),
        "fingerprint": _dataset_fingerprint(pairs),
    }
    if SPLIT_FILE.exists():
        try:
            saved = torch.load(SPLIT_FILE, map_location="cpu", weights_only=False)
        except TypeError:
            saved = torch.load(SPLIT_FILE, map_location="cpu")
        if isinstance(saved, dict) and all(saved.get(key) == value for key, value in expected.items()):
            indices = saved.get("indices")
            if isinstance(indices, dict):
                return indices
    permutation = torch.randperm(total, generator=torch.Generator().manual_seed(SPLIT_SEED)).tolist()
    indices = {
        "train": permutation[:train_count],
        "val": permutation[train_count : train_count + val_count],
        "test": permutation[train_count + val_count :],
    }
    torch.save({**expected, "indices": indices}, SPLIT_FILE)
    return indices


def _loader_kwargs(device: torch.device) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda" and PIN_MEMORY,
        "worker_init_fn": seed_worker if NUM_WORKERS > 0 else None,
        "persistent_workers": PERSISTENT_WORKERS and NUM_WORKERS > 0,
        "collate_fn": collate_paired_batch,
        "drop_last": False,
    }
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return kwargs


def make_inline_dataloaders(device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader]:
    pairs = validate_paired_samples(discover_paired_samples(resolve_data_root()))
    indices = load_or_create_split_indices(pairs)
    train_pool = NTIREPairedDataset(pairs, training=True, cube_key=HSI_KEY)
    eval_pool = NTIREPairedDataset(pairs, training=False, cube_key=HSI_KEY)
    train_dataset = Subset(train_pool, indices["train"])
    val_dataset = Subset(eval_pool, indices["val"])
    test_dataset = Subset(eval_pool, indices["test"])
    kwargs = _loader_kwargs(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        **kwargs,
    )
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs)
    print(f"Split manifest: {SPLIT_FILE}")
    print(f"Dataset split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    return train_loader, val_loader, test_loader


def make_repo_dataloaders(device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader]:
    common = dict(
        root_dir=str(DATA_ROOT),
        train_images=REPO_TRAIN_IMAGES,
        total_images=REPO_TOTAL_IMAGES,
        cube_key=HSI_KEY,
        download=REPO_DOWNLOAD_DATA,
        spectral_dir=REPO_SPECTRAL_DIR,
        rgb_dir=REPO_RGB_DIR,
        patch_size=int(TRAIN_PATCH_SIZE or 0),
        samples_per_image=REPO_SAMPLES_PER_IMAGE,
        augment=USE_GEOMETRIC_AUGMENTATION,
        hsi_scale=REPO_HSI_SCALE,
        clamp_hsi=CLIP_INPUTS_TO_UNIT_RANGE,
        num_bands=NUM_BANDS,
    )
    train_dataset = ARADDataset(train=True, **common)
    heldout_dataset = ARADDataset(train=False, **common)
    if REPO_TEST_USE_RANDOM_ARAD_LOADER:
        test_dataset, _ = load_random_arad1k_samples(
            root_dir=str(DATA_ROOT),
            num_samples=REPO_RANDOM_TEST_IMAGES,
            seed=VAL_SEED,
            total_images=REPO_RANDOM_TEST_TOTAL_IMAGES,
            cube_key=HSI_KEY,
            train_images=REPO_TRAIN_IMAGES,
            spectral_dir=REPO_SPECTRAL_DIR,
            rgb_dir=REPO_RGB_DIR,
            hsi_scale=REPO_HSI_SCALE,
        )
        val_dataset = heldout_dataset
    elif REPO_SPLIT_HELDOUT_FOR_TEST and len(heldout_dataset) >= 2:
        val_count = max(1, len(heldout_dataset) // 2)
        test_count = len(heldout_dataset) - val_count
        val_dataset, test_dataset = random_split(
            heldout_dataset,
            [val_count, test_count],
            generator=torch.Generator().manual_seed(SPLIT_SEED),
        )
    else:
        val_dataset = heldout_dataset
        test_dataset = heldout_dataset
        print("Warning: repo backend uses the same held-out samples for validation and test.")

    kwargs = _loader_kwargs(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        **kwargs,
    )
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs)
    print(f"Repo loader split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    return train_loader, val_loader, test_loader


def make_dataloaders(device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if DATA_BACKEND == "inline":
        return make_inline_dataloaders(device)
    if DATA_BACKEND == "repo":
        return make_repo_dataloaders(device)
    raise ValueError("DATA_BACKEND must be 'inline' or 'repo'")


# ==================================================
# EMA AND CHECKPOINTS
# ==================================================


class ExponentialMovingAverage:
    def __init__(self, module: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = deepcopy(module).eval()
        set_requires_grad(self.shadow, False)

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        source = module.state_dict()
        for key, value in self.shadow.state_dict().items():
            if value.is_floating_point():
                value.lerp_(source[key].detach(), 1.0 - self.decay)
            else:
                value.copy_(source[key])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow.state_dict()

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow.load_state_dict(state)


def checkpoint_paths(stage: int) -> Tuple[Path, Path, Path]:
    mapping = {
        1: (STAGE1_BEST_PATH, STAGE1_BEST_LOSS_PATH, STAGE1_LATEST_PATH),
        2: (STAGE2_BEST_PATH, STAGE2_BEST_LOSS_PATH, STAGE2_LATEST_PATH),
        3: (STAGE3_BEST_PATH, STAGE3_BEST_LOSS_PATH, STAGE3_LATEST_PATH),
    }
    if stage not in mapping:
        raise ValueError("stage must be 1, 2, or 3")
    return mapping[stage]


def save_checkpoint(
    path: Path,
    *,
    stage: int,
    epoch: int,
    model: LatentDiffusionRGB2HSI,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    best_val_mrae: float,
    best_val_loss: float,
    epochs_without_improvement: int,
    ema: Optional[ExponentialMovingAverage] = None,
) -> None:
    payload = {
        "stage": stage,
        "epoch": epoch,
        "model": model.state_dict(),
        "model_config": model.config.to_dict(),
        "best_val_mrae": best_val_mrae,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "ema_denoiser": ema.state_dict() if ema is not None else None,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Union[str, Path], device: torch.device) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Invalid checkpoint: {path}")
    return checkpoint


def configure_trainable_stage(model: LatentDiffusionRGB2HSI, stage: int) -> None:
    set_requires_grad(model.autoencoder, stage == 1)
    set_requires_grad(model.rgb_initializer, stage == 2)
    set_requires_grad(model.forward_operator, stage == 2)
    set_requires_grad(model.denoiser, stage == 3)
    if stage != 1:
        model.autoencoder.eval()
    if stage != 2:
        model.rgb_initializer.eval()
        model.forward_operator.eval()
    if stage != 3:
        model.denoiser.eval()


def build_training_model(stage: int, device: torch.device) -> LatentDiffusionRGB2HSI:
    model = LatentDiffusionRGB2HSI(make_model_config()).to(device)
    if stage == 2:
        checkpoint = load_checkpoint(STAGE1_TEACHER_CHECKPOINT, device)
        if int(checkpoint.get("stage", -1)) != 1:
            raise ValueError("STAGE1_TEACHER_CHECKPOINT must be a Stage-1 checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
    elif stage == 3:
        checkpoint = load_checkpoint(STAGE2_TEACHER_CHECKPOINT, device)
        if int(checkpoint.get("stage", -1)) != 2:
            raise ValueError("STAGE2_TEACHER_CHECKPOINT must be a Stage-2 checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
    configure_trainable_stage(model, stage)
    return model


def build_evaluation_model(checkpoint: Dict[str, Any], device: torch.device) -> LatentDiffusionRGB2HSI:
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = LatentDiffusionRGB2HSI(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if checkpoint.get("ema_denoiser") is not None:
        model.denoiser.load_state_dict(checkpoint["ema_denoiser"], strict=True)
    configure_trainable_stage(model, 0)
    model.eval()
    return model


# ==================================================
# STAGE LOSSES
# ==================================================


def spectral_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])


def autoencoder_objective(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    terms = {
        "l1": l1_loss(pred, target),
        "mrae": mrae(pred, target, eps=MRAE_EPS),
        "sam": sam(pred, target, eps=MRAE_EPS, degrees=False),
        "spectral_grad": spectral_gradient_loss(pred, target),
    }
    total = (
        AE_L1_WEIGHT * terms["l1"]
        + AE_MRAE_WEIGHT * terms["mrae"]
        + AE_SAM_WEIGHT * terms["sam"]
        + AE_SPECTRAL_GRAD_WEIGHT * terms["spectral_grad"]
    )
    return total, terms


def initializer_objective(
    model: LatentDiffusionRGB2HSI,
    rgb: torch.Tensor,
    hsi: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    with torch.no_grad():
        target_latent = model.normalize_latent(model.autoencoder.encode(hsi))
    pred_hsi, pred_latent = model.stage2(rgb)
    rgb_hat = model.forward_operator(hsi)
    terms = {
        "latent_l1": F.l1_loss(pred_latent, target_latent),
        "hsi_mrae": mrae(pred_hsi, hsi, eps=MRAE_EPS),
        "hsi_l1": l1_loss(pred_hsi, hsi),
        "hsi_sam": sam(pred_hsi, hsi, eps=MRAE_EPS, degrees=False),
        "rgb_l1": l1_loss(rgb_hat, rgb),
        "rgb_mse": mse_loss(rgb_hat, rgb),
        "rgb_ssim": 1.0 - ssim(rgb_hat, rgb, data_range=1.0, window_size=SSIM_WINDOW_SIZE),
        "operator_smooth": model.forward_operator.smoothness_regularizer(),
    }
    total = (
        INIT_LATENT_L1_WEIGHT * terms["latent_l1"]
        + INIT_HSI_MRAE_WEIGHT * terms["hsi_mrae"]
        + INIT_HSI_L1_WEIGHT * terms["hsi_l1"]
        + INIT_HSI_SAM_WEIGHT * terms["hsi_sam"]
        + FORWARD_RGB_L1_WEIGHT * terms["rgb_l1"]
        + FORWARD_RGB_MSE_WEIGHT * terms["rgb_mse"]
        + FORWARD_RGB_SSIM_WEIGHT * terms["rgb_ssim"]
        + FORWARD_SMOOTHNESS_WEIGHT * terms["operator_smooth"]
    )
    return total, terms, pred_hsi


def diffusion_objective(
    model: LatentDiffusionRGB2HSI,
    rgb: torch.Tensor,
    hsi: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    outputs = model.diffusion_training_outputs(rgb, hsi)
    decoded_mrae = torch.zeros((), device=hsi.device, dtype=hsi.dtype)
    predicted_hsi = model.stage2(rgb)[0].detach()
    threshold = int(model.config.diffusion_timesteps * DECODED_LOSS_MAX_T_FRACTION)
    mask = outputs["timesteps"] < threshold
    if mask.any():
        predicted_latent = outputs["rgb_latent"][mask] + outputs["predicted_residual"][mask]
        decoded = model.autoencoder.decode(model.denormalize_latent(predicted_latent))
        decoded_mrae = mrae(decoded, hsi[mask], eps=MRAE_EPS)
        # A training-only x0 prediction is used for progress metrics. It is not
        # the multi-step DDIM validation output.
        predicted_hsi = model.autoencoder.decode(
            model.denormalize_latent(outputs["rgb_latent"] + outputs["predicted_residual"])
        )
    terms = {
        "noise": F.mse_loss(outputs["predicted_noise"], outputs["noise"]),
        "residual_x0": F.l1_loss(outputs["predicted_residual"], outputs["clean_residual"]),
        "decoded_mrae": decoded_mrae,
    }
    total = (
        LAMBDA_NOISE * terms["noise"]
        + LAMBDA_RESIDUAL_X0 * terms["residual_x0"]
        + LAMBDA_DECODED_MRAE * terms["decoded_mrae"]
    )
    return total, terms, predicted_hsi


@torch.no_grad()
def compute_latent_statistics(
    model: LatentDiffusionRGB2HSI,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.autoencoder.eval()
    channel_sum = None
    channel_sumsq = None
    count = 0
    for index, batch in enumerate(loader):
        if LATENT_STATS_MAX_BATCHES is not None and index >= LATENT_STATS_MAX_BATCHES:
            break
        _, hsi, _, _ = unpack_batch(batch)
        hsi = hsi.to(device, non_blocking=device.type == "cuda").float().clamp(HSI_MIN, HSI_MAX)
        latent = model.autoencoder.encode(hsi).float()
        current_sum = latent.sum((0, 2, 3), keepdim=True)
        current_sumsq = latent.square().sum((0, 2, 3), keepdim=True)
        elements = latent.shape[0] * latent.shape[2] * latent.shape[3]
        channel_sum = current_sum if channel_sum is None else channel_sum + current_sum
        channel_sumsq = current_sumsq if channel_sumsq is None else channel_sumsq + current_sumsq
        count += elements
    if count == 0 or channel_sum is None or channel_sumsq is None:
        raise RuntimeError("Could not compute latent statistics")
    mean = channel_sum / count
    variance = (channel_sumsq / count - mean.square()).clamp_min(1e-8)
    return mean, variance.sqrt()


# ==================================================
# VALIDATION
# ==================================================


def deterministic_initial_noise(
    model: LatentDiffusionRGB2HSI,
    rgb: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    latent_h = rgb.shape[-2] // 4
    latent_w = rgb.shape[-1] // 4
    try:
        return torch.randn(
            (rgb.shape[0], model.config.latent_channels, latent_h, latent_w),
            device=rgb.device,
            dtype=rgb.dtype,
            generator=generator,
        )
    except TypeError:
        return torch.randn(
            (rgb.shape[0], model.config.latent_channels, latent_h, latent_w),
            device=rgb.device,
            dtype=rgb.dtype,
        )


def validate(
    model: LatentDiffusionRGB2HSI,
    loader: DataLoader,
    device: torch.device,
    stage: int,
    split_name: str = "Validation",
    max_images: Optional[int] = None,
    sampling_steps: Optional[int] = None,
    save_predictions: bool = False,
) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "mrae": 0.0, "rmse": 0.0, "psnr": 0.0, "sam": 0.0, "ssim": 0.0}
    count = 0
    total_batches = len(loader)
    started = time.perf_counter()
    prediction_dir = OUTPUT_DIR / f"{split_name.lower()}_stage{stage}_predictions"
    if save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    try:
        eval_generator = torch.Generator(device=device)
    except TypeError:
        eval_generator = torch.Generator()
    eval_generator.manual_seed(VAL_SEED)

    print(f"{split_name} started: {total_batches} batches", flush=True)
    for batch_index, batch in enumerate(loader):
        rgb, hsi, names, orig_hw = unpack_batch(batch)
        orig_hw_tensor = make_orig_hw_tensor(orig_hw, hsi)
        rgb = rgb.to(device, non_blocking=device.type == "cuda").float().clamp(0.0, 1.0)
        hsi = hsi.to(device, non_blocking=device.type == "cuda").float().clamp(HSI_MIN, HSI_MAX)

        if stage == 1:
            with torch.no_grad(), autocast_context(USE_AMP):
                pred_hsi, _ = model.stage1(hsi)
                loss, _ = autoencoder_objective(pred_hsi, hsi)
        elif stage == 2:
            with torch.no_grad(), autocast_context(USE_AMP):
                loss, _, pred_hsi = initializer_objective(model, rgb, hsi)
        elif stage == 3:
            initial_noise = deterministic_initial_noise(model, rgb, eval_generator)
            pred_hsi = model.reconstruct(
                rgb,
                sampling_steps=sampling_steps or SAMPLING_STEPS,
                guidance_scale=PHYSICS_GUIDANCE_SCALE,
                initial_noise=initial_noise,
            )
            with torch.no_grad():
                loss = mrae(pred_hsi, hsi, eps=MRAE_EPS)
        else:
            raise ValueError("stage must be 1, 2, or 3")

        for sample_index in range(pred_hsi.shape[0]):
            if max_images is not None and count >= max_images:
                break
            sample_pred, sample_hsi = crop_sample(pred_hsi.detach(), hsi, orig_hw_tensor, sample_index)
            metrics = compute_metrics(
                sample_pred.float(), sample_hsi.float(),
                mrae_eps=MRAE_EPS,
                ssim_window_size=SSIM_WINDOW_SIZE,
            )
            # Use per-image MRAE as the validation objective for checkpoint
            # selection in every stage, matching the NTIRE comparison metric.
            totals["loss"] += metrics["mrae"]
            for metric_name in TRAIN_DISPLAY_METRICS:
                totals[metric_name] += metrics[metric_name]
            if save_predictions:
                if isinstance(names, (list, tuple)) and sample_index < len(names):
                    sample_name = str(names[sample_index])
                else:
                    sample_name = f"sample_{count:04d}"
                cube = sample_pred.squeeze(0).cpu().permute(1, 2, 0).numpy().astype(np.float32)
                savemat(prediction_dir / f"{Path(sample_name).stem}.mat", {HSI_KEY: cube})
            count += 1

        completed = batch_index + 1
        if completed % PROGRESS_EVERY_N_BATCHES == 0 or completed == total_batches:
            running = {name: totals[name] / max(count, 1) for name in TRAIN_DISPLAY_METRICS}
            print(
                f"{split_name} batch {completed}/{total_batches} "
                f"| MRAE {running['mrae']:.6f} | RMSE {running['rmse']:.6f} "
                f"| PSNR {running['psnr']:.4f} | SAM {running['sam']:.4f} "
                f"| SSIM {running['ssim']:.6f} "
                f"| elapsed {(time.perf_counter()-started)/60:.1f} min",
                flush=True,
            )
        if max_images is not None and count >= max_images:
            break

    if count == 0:
        raise RuntimeError(f"{split_name} loader is empty")
    return {name: value / count for name, value in totals.items()}


# ==================================================
# TRAINING
# ==================================================


def stage_settings(stage: int) -> Tuple[int, float]:
    if stage == 1:
        return STAGE1_EPOCHS, STAGE1_LR
    if stage == 2:
        return STAGE2_EPOCHS, STAGE2_LR
    if stage == 3:
        return STAGE3_EPOCHS, STAGE3_LR
    raise ValueError("stage must be 1, 2, or 3")


def train(stage: int) -> None:
    set_seed(SEED)
    device = torch.device(DEVICE)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.empty_cache()

    train_loader, val_loader, test_loader = make_dataloaders(device)
    model = build_training_model(stage, device)

    if stage == 2 and RESUME_CHECKPOINT is None:
        mean, std = compute_latent_statistics(model, train_loader, device)
        model.set_latent_statistics(mean, std)
        print(
            f"Latent statistics set: mean range [{mean.min().item():.4f}, {mean.max().item():.4f}], "
            f"std range [{std.min().item():.4f}, {std.max().item():.4f}]"
        )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    epochs, learning_rate = stage_settings(stage)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=MIN_LR,
    )
    amp_enabled = USE_AMP and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)
    ema = ExponentialMovingAverage(model.denoiser, EMA_DECAY) if stage == 3 and USE_EMA else None
    if ema is not None:
        ema.shadow.to(device)

    start_epoch = 1
    best_val_mrae = math.inf
    best_val_loss = math.inf
    epochs_without_improvement = 0

    if RESUME_CHECKPOINT is not None:
        resume = load_checkpoint(RESUME_CHECKPOINT, device)
        if int(resume.get("stage", -1)) != stage:
            raise ValueError("RESUME_CHECKPOINT stage does not match --stage")
        model.load_state_dict(resume["model"], strict=True)
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        if "scheduler" in resume:
            scheduler.load_state_dict(resume["scheduler"])
        if ema is not None and resume.get("ema_denoiser") is not None:
            ema.load_state_dict(resume["ema_denoiser"])
        start_epoch = int(resume.get("epoch", 0)) + 1
        best_val_mrae = float(resume.get("best_val_mrae", math.inf))
        best_val_loss = float(resume.get("best_val_loss", math.inf))
        epochs_without_improvement = int(resume.get("epochs_without_improvement", 0))

    best_path, best_loss_path, latest_path = checkpoint_paths(stage)
    print("Execution mode: single GPU (no DataParallel/DDP)")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Stage: {stage}")
    print(f"Data backend: {DATA_BACKEND}")
    print(f"Train/val/test samples: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"Trainable parameters: {count_parameters(trainable_parameters):,}")
    print(f"Model configuration: {model.config.to_dict()}")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        print(f"\nEpoch {epoch}/{epochs} started ({len(train_loader)} batches)", flush=True)
        model.train()
        configure_trainable_stage(model, stage)
        if stage == 1:
            model.autoencoder.train()
        elif stage == 2:
            model.rgb_initializer.train()
            model.forward_operator.train()
        else:
            model.denoiser.train()

        running_total = 0.0
        running_terms: Dict[str, float] = {}
        running_metrics = {name: 0.0 for name in TRAIN_DISPLAY_METRICS}
        metric_sample_count = 0
        train_count = 0
        previous_batch_end = time.perf_counter()

        for batch_index, batch in enumerate(train_loader):
            batch_start = time.perf_counter()
            data_wait = batch_start - previous_batch_end
            rgb, hsi, names, orig_hw = unpack_batch(batch)
            orig_hw_tensor = make_orig_hw_tensor(orig_hw, hsi)
            if batch_index == 0:
                print(f"First batch | RGB {tuple(rgb.shape)} | HSI {tuple(hsi.shape)} | sample {names}")
            rgb = rgb.to(device, non_blocking=device.type == "cuda").float().clamp(0.0, 1.0)
            hsi = hsi.to(device, non_blocking=device.type == "cuda").float().clamp(HSI_MIN, HSI_MAX)
            batch_size = rgb.shape[0]
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(amp_enabled):
                if stage == 1:
                    pred_hsi, _ = model.stage1(hsi)
                    total_loss, terms = autoencoder_objective(pred_hsi, hsi)
                elif stage == 2:
                    total_loss, terms, pred_hsi = initializer_objective(model, rgb, hsi)
                else:
                    total_loss, terms, pred_hsi = diffusion_objective(model, rgb, hsi)

            scaler.scale(total_loss).backward()
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_parameters, GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model.denoiser)

            running_total += float(total_loss.detach()) * batch_size
            for key, value in terms.items():
                running_terms[key] = running_terms.get(key, 0.0) + float(value.detach()) * batch_size
            train_count += batch_size

            completed = batch_index + 1
            should_measure = completed % TRAIN_METRICS_EVERY_N_BATCHES == 0 or completed == len(train_loader)
            should_log = completed % PROGRESS_EVERY_N_BATCHES == 0 or completed == len(train_loader)
            if should_measure:
                metrics = compute_batch_display_metrics(pred_hsi.detach(), hsi, orig_hw_tensor)
                for name in TRAIN_DISPLAY_METRICS:
                    running_metrics[name] += metrics[name] * batch_size
                metric_sample_count += batch_size

            if should_log:
                average_metrics = {
                    name: running_metrics[name] / max(metric_sample_count, 1)
                    for name in TRAIN_DISPLAY_METRICS
                }
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    memory = (
                        f" | CUDA {torch.cuda.memory_allocated(device)/(1024**3):.2f}/"
                        f"{torch.cuda.memory_reserved(device)/(1024**3):.2f} GiB"
                    )
                else:
                    memory = ""
                print(
                    f"Epoch {epoch} batch {completed}/{len(train_loader)} "
                    f"| Loss {running_total/max(train_count,1):.6f} "
                    f"| MRAE {average_metrics['mrae']:.6f} "
                    f"| RMSE {average_metrics['rmse']:.6f} "
                    f"| PSNR {average_metrics['psnr']:.4f} "
                    f"| SAM {average_metrics['sam']:.4f} "
                    f"| SSIM {average_metrics['ssim']:.6f} "
                    f"| data {data_wait:.2f}s | step {time.perf_counter()-batch_start:.1f}s{memory}",
                    flush=True,
                )
            previous_batch_end = time.perf_counter()

        original_denoiser_state = None
        if ema is not None:
            original_denoiser_state = deepcopy(model.denoiser.state_dict())
            model.denoiser.load_state_dict(ema.state_dict(), strict=True)

        val_results = validate(
            model,
            val_loader,
            device,
            stage,
            split_name="Validation",
            max_images=DIFFUSION_VAL_MAX_IMAGES if stage == 3 else None,
            sampling_steps=DIFFUSION_VAL_SAMPLING_STEPS if stage == 3 else None,
        )

        if original_denoiser_state is not None:
            model.denoiser.load_state_dict(original_denoiser_state, strict=True)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        next_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{epochs} | Train loss {running_total/max(train_count,1):.6f} "
            f"| Val MRAE {val_results['mrae']:.6f} | Val RMSE {val_results['rmse']:.6f} "
            f"| Val PSNR {val_results['psnr']:.4f} | Val SAM {val_results['sam']:.4f} "
            f"| Val SSIM {val_results['ssim']:.6f} | LR {current_lr:.2e} "
            f"| Next LR {next_lr:.2e} | time {(time.perf_counter()-epoch_start)/60:.1f} min"
        )

        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            save_checkpoint(
                best_loss_path,
                stage=stage,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_mrae=best_val_mrae,
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
                ema=ema,
            )

        if val_results["mrae"] < best_val_mrae:
            best_val_mrae = val_results["mrae"]
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                stage=stage,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_mrae=best_val_mrae,
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
                ema=ema,
            )
            print(f"Saved best Stage-{stage} model (Val MRAE {best_val_mrae:.6f})")
        else:
            epochs_without_improvement += 1
            print(f"No validation MRAE improvement for {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs")

        save_checkpoint(
            latest_path,
            stage=stage,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_mrae=best_val_mrae,
            best_val_loss=best_val_loss,
            epochs_without_improvement=epochs_without_improvement,
            ema=ema,
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping. Best validation MRAE: {best_val_mrae:.6f}")
            break


# ==================================================
# EVALUATION
# ==================================================


def resolve_evaluation_checkpoint(stage: int) -> Path:
    if EVAL_CHECKPOINT is not None:
        selected = Path(EVAL_CHECKPOINT)
        if not selected.exists():
            raise FileNotFoundError(selected)
        return selected
    best, best_loss, latest = checkpoint_paths(stage)
    for candidate in (best, best_loss, latest):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No Stage-{stage} checkpoint found")


def evaluate(stage: int, split_name: str) -> None:
    set_seed(SEED)
    device = torch.device(DEVICE)
    _, val_loader, test_loader = make_dataloaders(device)
    loader = val_loader if split_name == "val" else test_loader
    selected_path = resolve_evaluation_checkpoint(stage)
    checkpoint = load_checkpoint(selected_path, device)
    if int(checkpoint.get("stage", -1)) != stage:
        raise ValueError(f"Checkpoint is Stage {checkpoint.get('stage')}, requested Stage {stage}")
    model = build_evaluation_model(checkpoint, device)
    results = validate(
        model,
        loader,
        device,
        stage,
        split_name=split_name.capitalize(),
        max_images=EVAL_MAX_IMAGES,
        sampling_steps=SAMPLING_STEPS if stage == 3 else None,
        save_predictions=SAVE_TEST_PREDICTIONS and split_name == "test",
    )
    print(f"Evaluated split: {split_name}")
    print(f"Evaluated samples: {len(loader.dataset)}")
    print(f"Evaluated checkpoint: {selected_path}")
    print(
        f"MRAE {results['mrae']:.6f} | RMSE {results['rmse']:.6f} "
        f"| SAM {results['sam']:.4f} | PSNR {results['psnr']:.4f} "
        f"| SSIM {results['ssim']:.6f}"
    )


# ==================================================
# MAIN
# ==================================================


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args.stage)
    elif args.mode == "val":
        evaluate(args.stage, "val")
    elif args.mode in {"test", "eval"}:
        evaluate(args.stage, "test")
    else:
        raise ValueError("mode must be train, val, test, or eval")


if __name__ == "__main__":
    main()
