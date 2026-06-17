#!/usr/bin/env python3
"""
data_processing.py

Project:
    Applying CNN to dMRI images:
    Automatic Identification and Classification of Aging Brain

Purpose:
    Convert TOPUP/EDDY-corrected diffusion MRI data into fixed-shape,
    multi-channel white-matter volumes that can later be used by a 3D CNN.

Main processing stages:
    1. Discover TOPUP/EDDY-corrected DWI files.
    2. Load DWI, b-values, and EDDY-rotated b-vectors.
    3. Build a DIPY gradient table and verify dimensions.
    4. Estimate a brain mask from all available b0 volumes.
    5. Fit a diffusion tensor using weighted least squares.
    6. Generate FA, MD, AD, and RD scalar maps.
    7. Affine-register FA to a common FA template.
    8. Apply the same affine transform to MD, AD, RD, and the mask.
    9. Apply a common template white-matter mask.
   10. Save fixed-shape CNN input arrays and quality-control figures.
   11. Create a dataset manifest containing age-bin labels.

Important:
    - This script ASSUMES TOPUP and EDDY have already been completed.
    - It does not perform tractography. Tractograms are not fixed-size image
      tensors and are not required for the first 3D-CNN baseline.
    - For final CNN training, use the generated .npz files and split subjects
      into train/validation/test sets before computing any dataset-level
      normalization statistics.

Expected ds000221-style paths:
    Raw dataset:
        <dataset_root>/participants.tsv
        <dataset_root>/sub-XXXXXX/ses-01/dwi/
            sub-XXXXXX_ses-01_dwi.bval

    Corrected derivative:
        <preproc_dir>/sub-XXXXXX/ses-01/dwi/
            dwi.nii.gz
            dwi.eddy_rotated_bvecs

Recommended template:
    $FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz

Example:
    python data_processing.py \
        --dataset-root /home/minnie1005/Intro_to_dMRI_workshop/data/ds000221 \
        --preproc-dir /home/minnie1005/Intro_to_dMRI_workshop/data/ds000221/derivatives/uncorrected_topup_eddy \
        --output-dir /home/minnie1005/brainHack_project/processed_data \
        --template auto \
        --subjects 010006

Process all discovered subjects:
    python data_processing.py \
        --dataset-root /home/minnie1005/Intro_to_dMRI_workshop/data/ds000221 \
        --preproc-dir /home/minnie1005/Intro_to_dMRI_workshop/data/ds000221/derivatives/uncorrected_topup_eddy \
        --output-dir /home/minnie1005/brainHack_project/processed_data \
        --template auto \
        --only-labeled
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_to_output

from dipy.align.imaffine import (
    AffineRegistration,
    MutualInformationMetric,
    transform_centers_of_mass,
)
from dipy.align.transforms import (
    AffineTransform3D,
    RigidTransform3D,
    TranslationTransform3D,
)
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst import dti
from dipy.segment.mask import median_otsu

try:
    from bids.layout import BIDSLayout
except ImportError:
    BIDSLayout = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable


SCRIPT_VERSION = "1.0.0"
CHANNEL_NAMES = ("FA", "MD", "AD", "RD")

# Fixed, subject-independent scaling used only for CNN-ready arrays.
# Raw NIfTI maps are also saved in physical units.
CNN_SCALING = {
    "FA": (0.0, 1.0),
    "MD": (0.0, 3.0e-3),
    "AD": (0.0, 4.0e-3),
    "RD": (0.0, 3.0e-3),
}


@dataclass(frozen=True)
class SubjectRecord:
    subject: str
    session: str
    dwi_path: Path
    bval_path: Path
    bvec_path: Path


@dataclass(frozen=True)
class TemplateBundle:
    source_path: Path
    image: nib.Nifti1Image
    data: np.ndarray
    affine: np.ndarray
    wm_mask: np.ndarray


def normalize_subject_label(value: str) -> str:
    """Convert 'sub-010006' or '010006' to '010006'."""
    value = str(value).strip()
    return value[4:] if value.startswith("sub-") else value


def normalize_session_label(value: str) -> str:
    """Convert 'ses-01' or '01' to '01'."""
    value = str(value).strip()
    return value[4:] if value.startswith("ses-") else value


def setup_logging(output_dir: Path, verbose: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s | %(levelname)s | %(message)s"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "data_processing.log", encoding="utf-8"),
    ]

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True,
    )


def resolve_template_path(template_arg: str) -> Path:
    """
    Resolve an FA template path.

    'auto' checks:
        1. $FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz
        2. Common Linux FSL installation paths
    """
    if template_arg.lower() != "auto":
        path = Path(template_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        return path

    candidates: list[Path] = []

    fsldir = os.environ.get("FSLDIR")
    if fsldir:
        candidates.append(
            Path(fsldir) / "data" / "standard" / "FMRIB58_FA_1mm.nii.gz"
        )

    candidates.extend(
        [
            Path("/usr/local/fsl/data/standard/FMRIB58_FA_1mm.nii.gz"),
            Path("/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz"),
            Path("/usr/share/fsl/6.0/data/standard/FMRIB58_FA_1mm.nii.gz"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "FMRIB58_FA_1mm.nii.gz was not found automatically. "
        "Install FSL or provide the file explicitly with:\n"
        "  --template /full/path/to/FMRIB58_FA_1mm.nii.gz\n"
        "For a native-space test only, add --native-only."
    )


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    header: Optional[nib.Nifti1Header] = None,
    dtype: np.dtype = np.float32,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_header = header.copy() if header is not None else None
    if output_header is not None:
        output_header.set_data_dtype(dtype)
    image = nib.Nifti1Image(data.astype(dtype), affine, header=output_header)
    nib.save(image, str(output_path))


def prepare_template(
    template_path: Path,
    output_dir: Path,
    voxel_size_mm: float,
    wm_threshold: float,
) -> TemplateBundle:
    """
    Load an FA template, convert FSL's x10000 scaling if needed,
    and resample it to an isotropic CNN-friendly grid.
    """
    logging.info("Loading FA template: %s", template_path)
    template_img = nib.load(str(template_path))

    resampled_img = resample_to_output(
        template_img,
        voxel_sizes=(voxel_size_mm, voxel_size_mm, voxel_size_mm),
        order=1,
    )

    template_data = np.asarray(resampled_img.get_fdata(), dtype=np.float32)
    template_data = np.nan_to_num(
        template_data, nan=0.0, posinf=0.0, neginf=0.0
    )

    # FMRIB58_FA_1mm is commonly stored with values scaled by 10000.
    if float(np.max(template_data)) > 2.0:
        logging.info("Detected scaled FA template; dividing intensities by 10000.")
        template_data /= 10000.0

    template_data = np.clip(template_data, 0.0, 1.0)
    wm_mask = template_data >= wm_threshold

    if int(wm_mask.sum()) < 1000:
        raise ValueError(
            f"Template white-matter mask is unexpectedly small: "
            f"{int(wm_mask.sum())} voxels. Check the template and threshold."
        )

    template_out_dir = output_dir / "template"
    template_out_dir.mkdir(parents=True, exist_ok=True)

    prepared_template_path = (
        template_out_dir
        / f"FMRIB58_FA_res-{voxel_size_mm:g}mm.nii.gz"
    )
    template_mask_path = (
        template_out_dir
        / f"FMRIB58_FA_res-{voxel_size_mm:g}mm_WMmask.nii.gz"
    )

    save_nifti(
        template_data,
        resampled_img.affine,
        prepared_template_path,
        resampled_img.header,
    )
    save_nifti(
        wm_mask.astype(np.uint8),
        resampled_img.affine,
        template_mask_path,
        resampled_img.header,
        dtype=np.uint8,
    )

    logging.info(
        "Prepared template shape=%s, voxel size=%.2f mm, WM voxels=%d",
        template_data.shape,
        voxel_size_mm,
        int(wm_mask.sum()),
    )

    prepared_img = nib.Nifti1Image(
        template_data,
        resampled_img.affine,
        header=resampled_img.header,
    )

    return TemplateBundle(
        source_path=template_path,
        image=prepared_img,
        data=template_data,
        affine=np.asarray(resampled_img.affine),
        wm_mask=wm_mask,
    )


def discover_subject_records(
    dataset_root: Path,
    preproc_dir: Path,
    requested_subjects: Optional[set[str]] = None,
) -> list[SubjectRecord]:
    """
    Discover ds000221-style corrected DWI files.

    The derivative generated in the course is not fully BIDS-compliant, so the
    corrected DWI and rotated b-vectors are found by directory structure.
    """
    if not preproc_dir.exists():
        raise FileNotFoundError(f"Preprocessed directory not found: {preproc_dir}")

    records: list[SubjectRecord] = []

    for subject_dir in sorted(preproc_dir.glob("sub-*")):
        if not subject_dir.is_dir():
            continue

        subject = normalize_subject_label(subject_dir.name)
        if requested_subjects and subject not in requested_subjects:
            continue

        for session_dir in sorted(subject_dir.glob("ses-*")):
            if not session_dir.is_dir():
                continue

            session = normalize_session_label(session_dir.name)
            dwi_dir = session_dir / "dwi"
            if not dwi_dir.exists():
                continue

            exact_dwi = dwi_dir / "dwi.nii.gz"
            if exact_dwi.exists():
                dwi_candidates = [exact_dwi]
            else:
                dwi_candidates = sorted(dwi_dir.glob("*_dwi.nii.gz"))

            if not dwi_candidates:
                logging.warning(
                    "No corrected DWI found for sub-%s ses-%s", subject, session
                )
                continue

            dwi_path = dwi_candidates[0]

            exact_bvec = dwi_dir / "dwi.eddy_rotated_bvecs"
            if exact_bvec.exists():
                bvec_candidates = [exact_bvec]
            else:
                bvec_candidates = sorted(dwi_dir.glob("*.eddy_rotated_bvecs"))

            if not bvec_candidates:
                logging.warning(
                    "No EDDY-rotated b-vector file found for sub-%s ses-%s",
                    subject,
                    session,
                )
                continue

            bvec_path = bvec_candidates[0]

            raw_dwi_dir = (
                dataset_root / f"sub-{subject}" / f"ses-{session}" / "dwi"
            )
            exact_bval = (
                raw_dwi_dir / f"sub-{subject}_ses-{session}_dwi.bval"
            )

            if exact_bval.exists():
                bval_candidates = [exact_bval]
            else:
                bval_candidates = sorted(raw_dwi_dir.glob("*.bval"))

            if not bval_candidates:
                logging.warning(
                    "No b-value file found for sub-%s ses-%s",
                    subject,
                    session,
                )
                continue

            records.append(
                SubjectRecord(
                    subject=subject,
                    session=session,
                    dwi_path=dwi_path.resolve(),
                    bval_path=bval_candidates[0].resolve(),
                    bvec_path=bvec_path.resolve(),
                )
            )

    return records


def parse_age_bin(value: Any) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Parse values such as '20-25' into low, high, midpoint."""
    if value is None or pd.isna(value):
        return None, None, None

    text = str(value).strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", text)
    if not match:
        return None, None, None

    low = float(match.group(1))
    high = float(match.group(2))
    return low, high, (low + high) / 2.0


def load_demographics(
    dataset_root: Path,
    young_max: float,
    old_min: float,
) -> dict[str, dict[str, Any]]:
    """
    Load participants.tsv and create conservative binary age groups.

    young:
        age-bin upper bound <= young_max
    old:
        age-bin lower bound >= old_min
    middle-age bins:
        excluded from binary classification
    """
    participants_path = dataset_root / "participants.tsv"
    if not participants_path.exists():
        logging.warning("participants.tsv not found: %s", participants_path)
        return {}

    participants = pd.read_csv(participants_path, sep="\t", dtype=str)
    if "participant_id" not in participants.columns:
        raise ValueError("participants.tsv does not contain participant_id.")

    age_columns = [
        column for column in participants.columns if "age" in column.lower()
    ]
    if not age_columns:
        raise ValueError(
            "No age column was found in participants.tsv. "
            f"Available columns: {list(participants.columns)}"
        )
    age_column = age_columns[0]

    sex_columns = [
        column
        for column in participants.columns
        if column.lower() in {"sex", "gender"}
    ]
    sex_column = sex_columns[0] if sex_columns else None

    demographics: dict[str, dict[str, Any]] = {}

    for _, row in participants.iterrows():
        subject = normalize_subject_label(row["participant_id"])
        age_bin = row.get(age_column)
        age_low, age_high, age_midpoint = parse_age_bin(age_bin)

        group: Optional[str] = None
        label: Optional[int] = None

        if age_low is not None and age_high is not None:
            if age_high <= young_max:
                group = "young"
                label = 0
            elif age_low >= old_min:
                group = "old"
                label = 1

        demographics[subject] = {
            "participant_id": f"sub-{subject}",
            "age_bin": None if pd.isna(age_bin) else str(age_bin),
            "age_low": age_low,
            "age_high": age_high,
            "age_midpoint": age_midpoint,
            "group": group,
            "label": label,
            "sex": (
                None
                if sex_column is None or pd.isna(row.get(sex_column))
                else str(row.get(sex_column))
            ),
        }

    logging.info(
        "Loaded demographics for %d participants using age column '%s'.",
        len(demographics),
        age_column,
    )
    return demographics


def load_and_prepare_dwi(
    record: SubjectRecord,
    max_bval: float,
    b0_threshold: float,
) -> tuple[
    nib.Nifti1Image,
    np.ndarray,
    Any,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load corrected DWI, gradient data, select DTI-compatible shells,
    estimate a brain mask, and calculate mean b0.
    """
    dwi_img = nib.load(str(record.dwi_path))
    if len(dwi_img.shape) != 4:
        raise ValueError(f"DWI must be 4D; got shape {dwi_img.shape}")

    bvals, bvecs = read_bvals_bvecs(
        str(record.bval_path),
        str(record.bvec_path),
    )
    bvals = np.asarray(bvals, dtype=np.float64)
    bvecs = np.asarray(bvecs, dtype=np.float64)

    n_volumes = int(dwi_img.shape[-1])
    if n_volumes != len(bvals) or n_volumes != len(bvecs):
        raise ValueError(
            "DWI/gradient dimension mismatch: "
            f"DWI volumes={n_volumes}, bvals={len(bvals)}, bvecs={len(bvecs)}"
        )

    is_b0 = bvals <= b0_threshold
    selected = is_b0 | (bvals <= max_bval)

    # A diffusion tensor requires at least six independent diffusion-weighted
    # measurements. If the configured shell selection is too restrictive,
    # retain all volumes instead of silently producing an invalid fit.
    if int(np.sum(selected & ~is_b0)) < 6:
        logging.warning(
            "sub-%s ses-%s has fewer than 6 non-b0 volumes at b<=%.1f. "
            "Using all volumes for tensor fitting.",
            record.subject,
            record.session,
            max_bval,
        )
        selected = np.ones_like(bvals, dtype=bool)

    selected_bvals = bvals[selected]
    selected_bvecs = bvecs[selected]

    gtab = gradient_table(
        bvals=selected_bvals,
        bvecs=selected_bvecs,
        b0_threshold=b0_threshold,
    )

    b0_indices = np.flatnonzero(gtab.b0s_mask)
    if len(b0_indices) == 0:
        raise ValueError("No b0 volume was detected.")

    full_dwi_data = np.asarray(dwi_img.dataobj, dtype=np.float32)
    dwi_data = full_dwi_data[..., selected]
    dwi_data = np.nan_to_num(
        dwi_data, nan=0.0, posinf=0.0, neginf=0.0
    )

    masked_data, brain_mask = median_otsu(
        dwi_data,
        vol_idx=b0_indices.tolist(),
        median_radius=4,
        numpass=4,
        dilate=1,
    )

    brain_mask = brain_mask.astype(bool)
    if int(brain_mask.sum()) < 1000:
        raise ValueError(
            f"Brain mask is unexpectedly small: {int(brain_mask.sum())} voxels."
        )

    mean_b0 = np.mean(masked_data[..., b0_indices], axis=-1).astype(np.float32)

    return (
        dwi_img,
        masked_data.astype(np.float32),
        gtab,
        brain_mask,
        mean_b0,
        selected_bvals,
    )


def fit_dti_maps(
    dwi_data: np.ndarray,
    gtab: Any,
    brain_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit DTI with WLS and return scalar maps in physical units."""
    logging.info("Fitting diffusion tensor using WLS.")
    tensor_model = dti.TensorModel(gtab, fit_method="WLS")
    tensor_fit = tensor_model.fit(dwi_data, mask=brain_mask)

    # Negative eigenvalues can appear because of noise or imperfect correction.
    evals = np.clip(np.asarray(tensor_fit.evals), 0.0, None)

    fa = dti.fractional_anisotropy(evals)
    md = dti.mean_diffusivity(evals)
    ad = dti.axial_diffusivity(evals)
    rd = dti.radial_diffusivity(evals)

    maps = {
        "FA": np.clip(
            np.nan_to_num(fa, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            1.0,
        ).astype(np.float32),
        "MD": np.nan_to_num(
            md, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32),
        "AD": np.nan_to_num(
            ad, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32),
        "RD": np.nan_to_num(
            rd, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32),
    }

    for name in ("MD", "AD", "RD"):
        maps[name] = np.clip(maps[name], 0.0, None)

    return maps


def calculate_affine_registration(
    moving_fa: np.ndarray,
    moving_affine: np.ndarray,
    static_fa: np.ndarray,
    static_affine: np.ndarray,
    sampling_proportion: float,
) -> Any:
    """
    Progressive center-of-mass -> translation -> rigid -> affine registration.

    Returns the final DIPY AffineMap so the same transform can be applied to
    every scalar map from the same subject.
    """
    center_of_mass = transform_centers_of_mass(
        static_fa,
        static_affine,
        moving_fa,
        moving_affine,
    )

    metric = MutualInformationMetric(
        nbins=32,
        sampling_proportion=sampling_proportion,
    )
    affine_registration = AffineRegistration(
        metric=metric,
        level_iters=[1000, 100, 10],
        sigmas=[3.0, 1.0, 0.0],
        factors=[4, 2, 1],
    )

    translation = affine_registration.optimize(
        static_fa,
        moving_fa,
        TranslationTransform3D(),
        None,
        static_grid2world=static_affine,
        moving_grid2world=moving_affine,
        starting_affine=center_of_mass.affine,
    )

    rigid = affine_registration.optimize(
        static_fa,
        moving_fa,
        RigidTransform3D(),
        None,
        static_grid2world=static_affine,
        moving_grid2world=moving_affine,
        starting_affine=translation.affine,
    )

    affine_map = affine_registration.optimize(
        static_fa,
        moving_fa,
        AffineTransform3D(),
        None,
        static_grid2world=static_affine,
        moving_grid2world=moving_affine,
        starting_affine=rigid.affine,
    )

    return affine_map


def transform_scalar_maps(
    scalar_maps: dict[str, np.ndarray],
    brain_mask: np.ndarray,
    affine_map: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Apply one FA-derived affine transform to all subject scalar maps."""
    registered: dict[str, np.ndarray] = {}

    for name, data in scalar_maps.items():
        transformed = affine_map.transform(data, interpolation="linear")
        transformed = np.nan_to_num(
            transformed, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

        if name == "FA":
            transformed = np.clip(transformed, 0.0, 1.0)
        else:
            transformed = np.clip(transformed, 0.0, None)

        registered[name] = transformed

    registered_mask = affine_map.transform(
        brain_mask.astype(np.float32),
        interpolation="nearest",
    )
    registered_mask = registered_mask >= 0.5

    return registered, registered_mask


def create_cnn_tensor(
    registered_maps: dict[str, np.ndarray],
    common_wm_mask: np.ndarray,
) -> np.ndarray:
    """
    Create a channels-first tensor: [FA, MD, AD, RD, X, Y, Z].

    Fixed scaling preserves between-subject intensity differences. It avoids
    subject-wise z-scoring, which could remove global age-related differences.
    """
    channels: list[np.ndarray] = []

    for name in CHANNEL_NAMES:
        lower, upper = CNN_SCALING[name]
        data = np.clip(registered_maps[name], lower, upper)
        scaled = (data - lower) / (upper - lower)
        scaled = scaled * common_wm_mask
        channels.append(scaled.astype(np.float32))

    return np.stack(channels, axis=0).astype(np.float32)


def registration_correlation(
    registered_fa: np.ndarray,
    template_fa: np.ndarray,
    wm_mask: np.ndarray,
) -> Optional[float]:
    valid = (
        wm_mask
        & np.isfinite(registered_fa)
        & np.isfinite(template_fa)
        & (registered_fa > 0)
    )
    if int(valid.sum()) < 100:
        return None

    moving_values = registered_fa[valid].astype(np.float64)
    static_values = template_fa[valid].astype(np.float64)

    if np.std(moving_values) == 0 or np.std(static_values) == 0:
        return None

    return float(np.corrcoef(moving_values, static_values)[0, 1])


def save_qc_figure(
    output_path: Path,
    mean_b0: np.ndarray,
    native_fa: np.ndarray,
    native_mask: np.ndarray,
    template_fa: np.ndarray,
    registered_fa: np.ndarray,
    common_wm_mask: np.ndarray,
    registration_corr: Optional[float],
) -> None:
    """Save a compact QC image for brain masking and affine registration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    native_z = mean_b0.shape[2] // 2
    template_z = template_fa.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(
        mean_b0[:, :, native_z].T,
        cmap="gray",
        origin="lower",
    )
    axes[0, 0].set_title("Native mean b0")

    axes[0, 1].imshow(
        native_fa[:, :, native_z].T,
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    axes[0, 1].set_title("Native FA")

    axes[0, 2].imshow(
        mean_b0[:, :, native_z].T,
        cmap="gray",
        origin="lower",
    )
    axes[0, 2].imshow(
        native_mask[:, :, native_z].T,
        cmap="autumn",
        origin="lower",
        alpha=0.30,
    )
    axes[0, 2].set_title("Native brain mask")

    axes[1, 0].imshow(
        template_fa[:, :, template_z].T,
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title("FA template")

    axes[1, 1].imshow(
        registered_fa[:, :, template_z].T,
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    corr_text = "n/a" if registration_corr is None else f"{registration_corr:.3f}"
    axes[1, 1].set_title(f"Registered FA\nWM correlation={corr_text}")

    axes[1, 2].imshow(
        template_fa[:, :, template_z].T,
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    axes[1, 2].imshow(
        registered_fa[:, :, template_z].T,
        cmap="hot",
        origin="lower",
        vmin=0,
        vmax=1,
        alpha=0.45,
    )
    axes[1, 2].contour(
        common_wm_mask[:, :, template_z].T.astype(np.uint8),
        levels=[0.5],
        linewidths=0.6,
    )
    axes[1, 2].set_title("Template + registered FA + WM mask")

    for axis in axes.ravel():
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def subject_output_paths(
    output_dir: Path,
    subject: str,
    session: str,
) -> dict[str, Path]:
    prefix = f"sub-{subject}_ses-{session}"
    subject_root = output_dir / f"sub-{subject}" / f"ses-{session}"

    return {
        "root": subject_root,
        "native_dir": subject_root / "native",
        "common_dir": subject_root / "common_space",
        "cnn_dir": subject_root / "cnn",
        "qc_dir": subject_root / "qc",
        "metadata": subject_root / "cnn" / f"{prefix}_metadata.json",
        "npz": subject_root / "cnn" / f"{prefix}_dti_cnn_input.npz",
        "qc": subject_root / "qc" / f"{prefix}_qc.png",
    }


def process_subject(
    record: SubjectRecord,
    output_dir: Path,
    demographics: dict[str, dict[str, Any]],
    template: Optional[TemplateBundle],
    max_bval: float,
    b0_threshold: float,
    sampling_proportion: float,
    overwrite: bool,
) -> dict[str, Any]:
    paths = subject_output_paths(
        output_dir,
        record.subject,
        record.session,
    )

    for key in ("native_dir", "common_dir", "cnn_dir", "qc_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    if template is not None and paths["npz"].exists() and not overwrite:
        logging.info(
            "Skipping existing CNN input: sub-%s ses-%s",
            record.subject,
            record.session,
        )
        metadata: dict[str, Any] = {}
        if paths["metadata"].exists():
            with paths["metadata"].open("r", encoding="utf-8") as file:
                metadata = json.load(file)
        metadata["status"] = "skipped_existing"
        return metadata

    participant_info = demographics.get(
        record.subject,
        {
            "participant_id": f"sub-{record.subject}",
            "age_bin": None,
            "age_low": None,
            "age_high": None,
            "age_midpoint": None,
            "group": None,
            "label": None,
            "sex": None,
        },
    )

    logging.info(
        "Processing sub-%s ses-%s", record.subject, record.session
    )

    (
        dwi_img,
        dwi_data,
        gtab,
        brain_mask,
        mean_b0,
        selected_bvals,
    ) = load_and_prepare_dwi(
        record,
        max_bval=max_bval,
        b0_threshold=b0_threshold,
    )

    scalar_maps = fit_dti_maps(
        dwi_data=dwi_data,
        gtab=gtab,
        brain_mask=brain_mask,
    )

    prefix = f"sub-{record.subject}_ses-{record.session}"
    native_header = dwi_img.header.copy()

    save_nifti(
        mean_b0,
        dwi_img.affine,
        paths["native_dir"] / f"{prefix}_space-native_mean-b0.nii.gz",
        native_header,
    )
    save_nifti(
        brain_mask.astype(np.uint8),
        dwi_img.affine,
        paths["native_dir"] / f"{prefix}_space-native_brainmask.nii.gz",
        native_header,
        dtype=np.uint8,
    )

    native_map_paths: dict[str, str] = {}
    for name, data in scalar_maps.items():
        output_path = (
            paths["native_dir"]
            / f"{prefix}_space-native_desc-WLS_{name}.nii.gz"
        )
        save_nifti(data, dwi_img.affine, output_path, native_header)
        native_map_paths[name] = str(output_path)

    result: dict[str, Any] = {
        "status": "native_complete" if template is None else "success",
        "participant_id": f"sub-{record.subject}",
        "subject": record.subject,
        "session": record.session,
        **participant_info,
        "dwi_path": str(record.dwi_path),
        "bval_path": str(record.bval_path),
        "bvec_path": str(record.bvec_path),
        "selected_volume_count": int(len(selected_bvals)),
        "selected_bval_min": float(np.min(selected_bvals)),
        "selected_bval_max": float(np.max(selected_bvals)),
        "native_shape": list(dwi_img.shape[:3]),
        "native_map_paths": native_map_paths,
        "cnn_path": None,
        "qc_path": None,
        "registration_correlation": None,
    }

    if template is None:
        with paths["metadata"].open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)
        return result

    affine_map = calculate_affine_registration(
        moving_fa=scalar_maps["FA"],
        moving_affine=np.asarray(dwi_img.affine),
        static_fa=template.data,
        static_affine=template.affine,
        sampling_proportion=sampling_proportion,
    )

    registered_maps, registered_brain_mask = transform_scalar_maps(
        scalar_maps=scalar_maps,
        brain_mask=brain_mask,
        affine_map=affine_map,
    )

    np.savetxt(
        paths["common_dir"] / f"{prefix}_from-native_to-common_affine.txt",
        affine_map.affine,
        fmt="%.10f",
    )

    registered_map_paths: dict[str, str] = {}
    for name, data in registered_maps.items():
        output_path = (
            paths["common_dir"]
            / f"{prefix}_space-common_desc-WLS_{name}.nii.gz"
        )
        save_nifti(
            data,
            template.affine,
            output_path,
            template.image.header,
        )
        registered_map_paths[name] = str(output_path)

    save_nifti(
        registered_brain_mask.astype(np.uint8),
        template.affine,
        paths["common_dir"]
        / f"{prefix}_space-common_brainmask.nii.gz",
        template.image.header,
        dtype=np.uint8,
    )

    cnn_tensor = create_cnn_tensor(
        registered_maps=registered_maps,
        common_wm_mask=template.wm_mask,
    )

    corr = registration_correlation(
        registered_fa=registered_maps["FA"],
        template_fa=template.data,
        wm_mask=template.wm_mask,
    )

    np.savez_compressed(
        paths["npz"],
        x=cnn_tensor,
        mask=template.wm_mask.astype(np.uint8),
        affine=template.affine.astype(np.float64),
        channel_names=np.asarray(CHANNEL_NAMES),
        subject=np.asarray(record.subject),
        session=np.asarray(record.session),
        group=np.asarray(
            "" if participant_info.get("group") is None
            else participant_info["group"]
        ),
        label=np.asarray(
            -1 if participant_info.get("label") is None
            else int(participant_info["label"]),
            dtype=np.int64,
        ),
        age_bin=np.asarray(
            "" if participant_info.get("age_bin") is None
            else participant_info["age_bin"]
        ),
        age_midpoint=np.asarray(
            np.nan
            if participant_info.get("age_midpoint") is None
            else float(participant_info["age_midpoint"]),
            dtype=np.float32,
        ),
    )

    save_qc_figure(
        output_path=paths["qc"],
        mean_b0=mean_b0,
        native_fa=scalar_maps["FA"],
        native_mask=brain_mask,
        template_fa=template.data,
        registered_fa=registered_maps["FA"],
        common_wm_mask=template.wm_mask,
        registration_corr=corr,
    )

    result.update(
        {
            "common_shape": list(template.data.shape),
            "registered_map_paths": registered_map_paths,
            "cnn_path": str(paths["npz"]),
            "qc_path": str(paths["qc"]),
            "registration_correlation": corr,
            "channel_names": list(CHANNEL_NAMES),
            "cnn_scaling": {
                name: {
                    "minimum": CNN_SCALING[name][0],
                    "maximum": CNN_SCALING[name][1],
                }
                for name in CHANNEL_NAMES
            },
            "template_source": str(template.source_path),
            "template_shape": list(template.data.shape),
            "template_affine": template.affine.tolist(),
            "common_wm_voxels": int(template.wm_mask.sum()),
        }
    )

    with paths["metadata"].open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    logging.info(
        "Completed sub-%s ses-%s | CNN shape=%s | registration corr=%s",
        record.subject,
        record.session,
        cnn_tensor.shape,
        "n/a" if corr is None else f"{corr:.3f}",
    )

    return result


def write_dataset_description(output_dir: Path) -> None:
    description = {
        "Name": "Aging dMRI CNN preprocessing derivatives",
        "BIDSVersion": "1.11.1",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "data_processing.py",
                "Version": SCRIPT_VERSION,
                "Description": (
                    "DTI scalar-map generation, affine FA-template registration, "
                    "common white-matter masking, and CNN tensor export."
                ),
            }
        ],
    }

    with (output_dir / "dataset_description.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(description, file, indent=2, ensure_ascii=False)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create fixed-shape FA/MD/AD/RD white-matter tensors from "
            "TOPUP/EDDY-corrected dMRI for an aging-brain CNN project."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root of the raw BIDS dataset, for example .../data/ds000221",
    )
    parser.add_argument(
        "--preproc-dir",
        type=Path,
        required=True,
        help=(
            "TOPUP/EDDY derivative directory, for example "
            ".../derivatives/uncorrected_topup_eddy"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which processed outputs will be written.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="auto",
        help=(
            "FA template path, or 'auto' to locate FMRIB58_FA_1mm through FSL."
        ),
    )
    parser.add_argument(
        "--native-only",
        action="store_true",
        help=(
            "Generate native-space DTI maps only. No fixed-shape CNN arrays "
            "will be produced."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help=(
            "Optional subject labels, with or without 'sub-'. "
            "Omit to process every discovered subject."
        ),
    )
    parser.add_argument(
        "--only-labeled",
        action="store_true",
        help=(
            "Process only subjects assigned to the configured young/old groups."
        ),
    )
    parser.add_argument(
        "--young-max",
        type=float,
        default=35.0,
        help=(
            "A subject is young when the upper limit of the age bin is <= this."
        ),
    )
    parser.add_argument(
        "--old-min",
        type=float,
        default=55.0,
        help=(
            "A subject is old when the lower limit of the age bin is >= this."
        ),
    )
    parser.add_argument(
        "--max-bval",
        type=float,
        default=1500.0,
        help=(
            "Maximum nonzero b-value included in DTI fitting. "
            "b0 volumes are always retained."
        ),
    )
    parser.add_argument(
        "--b0-threshold",
        type=float,
        default=50.0,
        help="Volumes with b-values <= this are treated as b0.",
    )
    parser.add_argument(
        "--template-voxel-size",
        type=float,
        default=2.0,
        help=(
            "Isotropic template resolution in millimeters. "
            "2 mm is substantially more CNN-friendly than 1 mm."
        ),
    )
    parser.add_argument(
        "--wm-threshold",
        type=float,
        default=0.20,
        help="FA threshold used to construct the common template WM mask.",
    )
    parser.add_argument(
        "--registration-sampling",
        type=float,
        default=0.20,
        help=(
            "Proportion of voxels sampled for mutual-information registration."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing subject CNN outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    preproc_dir = args.preproc_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    setup_logging(output_dir=output_dir, verbose=args.verbose)
    write_dataset_description(output_dir)

    logging.info("data_processing.py version %s", SCRIPT_VERSION)
    logging.info("Dataset root: %s", dataset_root)
    logging.info("Preprocessed DWI directory: %s", preproc_dir)
    logging.info("Output directory: %s", output_dir)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # PyBIDS is used here to verify/index the raw BIDS tree. Course-generated
    # derivatives are discovered separately because their filenames are not
    # fully BIDS-compliant.
    if BIDSLayout is not None:
        try:
            raw_layout = BIDSLayout(
                str(dataset_root),
                validate=False,
                derivatives=False,
            )
            logging.info(
                "PyBIDS indexed %d raw-dataset subjects.",
                len(raw_layout.get_subjects()),
            )
        except Exception as error:
            logging.warning("PyBIDS indexing warning: %s", error)
    else:
        logging.warning(
            "PyBIDS is not installed. File discovery will still continue."
        )

    demographics = load_demographics(
        dataset_root=dataset_root,
        young_max=args.young_max,
        old_min=args.old_min,
    )

    requested_subjects = None
    if args.subjects:
        requested_subjects = {
            normalize_subject_label(subject) for subject in args.subjects
        }

    records = discover_subject_records(
        dataset_root=dataset_root,
        preproc_dir=preproc_dir,
        requested_subjects=requested_subjects,
    )

    if args.only_labeled:
        records = [
            record
            for record in records
            if demographics.get(record.subject, {}).get("label") in {0, 1}
        ]

    if not records:
        raise RuntimeError(
            "No processable subjects were discovered. Check --dataset-root, "
            "--preproc-dir, subject labels, and file names."
        )

    logging.info("Discovered %d subject/session records.", len(records))

    template: Optional[TemplateBundle] = None
    if not args.native_only:
        template_path = resolve_template_path(args.template)
        template = prepare_template(
            template_path=template_path,
            output_dir=output_dir,
            voxel_size_mm=args.template_voxel_size,
            wm_threshold=args.wm_threshold,
        )
    else:
        logging.warning(
            "Native-only mode enabled: fixed-shape CNN .npz files will not "
            "be generated."
        )

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for record in tqdm(records, desc="Processing dMRI subjects"):
        try:
            result = process_subject(
                record=record,
                output_dir=output_dir,
                demographics=demographics,
                template=template,
                max_bval=args.max_bval,
                b0_threshold=args.b0_threshold,
                sampling_proportion=args.registration_sampling,
                overwrite=args.overwrite,
            )
            results.append(result)
        except Exception as error:
            logging.error(
                "Failed sub-%s ses-%s: %s",
                record.subject,
                record.session,
                error,
            )
            logging.debug(traceback.format_exc())
            errors.append(
                {
                    "participant_id": f"sub-{record.subject}",
                    "subject": record.subject,
                    "session": record.session,
                    "status": "failed",
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )

    manifest_path = output_dir / "dataset_manifest.csv"
    if results:
        manifest = pd.DataFrame(results)

        preferred_columns = [
            "participant_id",
            "subject",
            "session",
            "age_bin",
            "age_midpoint",
            "sex",
            "group",
            "label",
            "status",
            "cnn_path",
            "qc_path",
            "registration_correlation",
            "dwi_path",
            "bval_path",
            "bvec_path",
        ]
        additional_scalar_columns = [
            "selected_volume_count",
            "selected_bval_min",
            "selected_bval_max",
            "native_shape",
            "common_shape",
            "template_source",
        ]
        ordered_columns = [
            column
            for column in preferred_columns + additional_scalar_columns
            if column in manifest.columns
        ]
        manifest = manifest[ordered_columns]
        manifest.to_csv(manifest_path, index=False)

    error_path = output_dir / "processing_errors.csv"
    if errors:
        pd.DataFrame(errors).to_csv(error_path, index=False)
    elif error_path.exists():
        error_path.unlink()

    logging.info(
        "Finished. Success/skipped=%d, failed=%d",
        len(results),
        len(errors),
    )
    logging.info("Manifest: %s", manifest_path)

    if template is not None:
        logging.info(
            "CNN input format: x.shape = (4, %d, %d, %d), "
            "channel order = %s",
            *template.data.shape,
            CHANNEL_NAMES,
        )

    return 1 if errors and not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
