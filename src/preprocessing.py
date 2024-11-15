"""
Data loading and preprocessing pipeline for BCI Competition IV 2a.

Pipeline per subject
--------------------
1. Load raw GDF via MNE (mne.io.read_raw_gdf)
2. Bandpass filter [0.5–100 Hz] + notch at 50 Hz  (FIR design)
3. Pick 5 motor-cortex electrodes: Fz, C3, Cz, C4, Oz
4. Extract epochs aligned to cue onset (tmin=0, tmax=4 s, no baseline)
5. Sub-sample each trial: 1000 → 375 time points (scipy.signal.resample)
6. Apply Morlet CWT per channel: scales 1–50 → (50, 375) scalogram
7. Stack 5 channels → (50, 375, 5) per trial
8. Normalise dataset to [-1, 1] (min-max, for GAN compatibility)

Output shape: (n_trials, 50, 375, 5)
"""

from __future__ import annotations

import numpy as np
import mne
import pywt
from scipy.signal import resample
from pathlib import Path
from typing import Dict, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATASET_DIR, ELECTRODE_NAMES, TMIN, TMAX,
    N_TIME_SUBSAMPLE, N_CWT_SCALES, CWT_WAVELET, CWT_SCALES,
    EVENT_IDS, TRIAL_SHAPE, N_CLASSES,
)

mne.set_log_level("WARNING")

_SCALES = np.array(CWT_SCALES, dtype=np.float64)


def gdf_path(subject: int, session: str = "T") -> Path:
    """Return path to A0<subject><session>.gdf."""
    return DATASET_DIR / f"A{subject:02d}{session}.gdf"


def load_raw(subject: int, session: str = "T") -> mne.io.BaseRaw:
    path = gdf_path(subject, session)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    raw = mne.io.read_raw_gdf(str(path), preload=True, verbose=False)
    raw.filter(l_freq=0.5, h_freq=100.0, method="fir", verbose=False)
    raw.notch_filter(freqs=50.0, method="fir", verbose=False)

    # BCI Competition IV 2a: 22 EEG channels + 3 EOG.
    # MNE renames duplicate bare "EEG" channels to EEG-0 … EEG-16, so the
    # standard names (EEG-Fz, EEG-C3, EEG-Cz, EEG-C4, EEG-Oz) may not all
    # be present.  Pick by fixed index within the EEG channels instead.
    #
    # BCI IV 2a channel order (0-based within EEG-only channels):
    #   0=Fz, 1=FC3, 2=FC1, 3=FCz, 4=FC2, 5=FC4,
    #   6=C5,  7=C3,  8=C1,  9=Cz, 10=C2, 11=C4,
    #  12=C6, 13=CP3, 14=CP1, 15=CPz, 16=CP2, 17=CP4,
    #  18=P1, 19=Pz,  20=P2,  21=Oz
    EEG_TARGET_INDICES = [0, 7, 9, 11, 21]   # Fz, C3, Cz, C4, Oz

    missing = set(ELECTRODE_NAMES) - set(raw.ch_names)
    if not missing:
        raw.pick_channels(ELECTRODE_NAMES, ordered=True)
    else:
        eeg_chs = [ch for ch in raw.ch_names if ch.startswith("EEG")]
        if len(eeg_chs) < 22:
            raise RuntimeError(
                f"Expected ≥22 EEG channels, found {len(eeg_chs)}.\n"
                f"Available: {raw.ch_names}"
            )
        selected = [eeg_chs[i] for i in EEG_TARGET_INDICES]
        raw.pick_channels(selected, ordered=True)
        rename_map = dict(zip(selected, ELECTRODE_NAMES))
        raw.rename_channels(rename_map)

    return raw


def extract_epochs(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract MI epochs from a filtered raw recording.

    Event codes (Table 2 of BCI Competition IV 2a data description):
      769 → class 0  Left hand
      770 → class 1  Right hand
      771 → class 2  Both feet
      772 → class 3  Tongue
      1023           Rejected / artifact trial — excluded from output

    Returns
    -------
    X : ndarray  (n_trials, n_channels, n_times)   raw EEG in µV
    y : ndarray  (n_trials,)                        0-indexed class labels
    """
    events, event_id_map = mne.events_from_annotations(raw, verbose=False)

    mi_event_id = {
        k: v for k, v in event_id_map.items()
        if int(k) in EVENT_IDS
    }
    if not mi_event_id:
        raise RuntimeError(
            "No MI class events (769-772) found in annotations.\n"
            f"Available annotations: {list(event_id_map.keys())}"
        )

    artifact_codes = {v for k, v in event_id_map.items() if int(k) == 1023}
    artifact_positions: set[int] = set()
    if artifact_codes:
        for code in artifact_codes:
            positions = events[events[:, 2] == code, 0]
            artifact_positions.update(int(p) for p in positions)

    mi_mask = np.isin(events[:, 2], list(mi_event_id.values()))
    mi_evs  = events[mi_mask]

    if artifact_positions:
        clean_mask = ~np.isin(mi_evs[:, 0], list(artifact_positions))
        n_removed  = int(np.sum(~clean_mask))
        mi_evs     = mi_evs[clean_mask]
        if n_removed:
            print(f"    ↳ Removed {n_removed} artifact-marked trial(s)")

    _, uniq = np.unique(mi_evs[:, 0], return_index=True)
    mi_evs  = mi_evs[uniq]

    epochs = mne.Epochs(
        raw, mi_evs,
        event_id=mi_event_id,
        tmin=TMIN, tmax=TMAX,
        baseline=None,
        preload=True,
        verbose=False,
    )
    epochs.drop_bad(verbose=False)

    X = epochs.get_data()                               

    inv_map = {v: int(k) for k, v in mi_event_id.items()}
    y = np.array([EVENT_IDS[inv_map[code]] for code in epochs.events[:, 2]])

    return X, y


def cwt_trial(trial: np.ndarray) -> np.ndarray:
    """
    Compute Morlet CWT for a single trial.

    Parameters
    ----------
    trial : ndarray  (n_channels, n_times)

    Returns
    -------
    cwt_image : ndarray  (50, 375, 5)   float32
    """
    n_channels, _ = trial.shape
    trial_sub = resample(trial, N_TIME_SUBSAMPLE, axis=-1)  # (C, 375)

    channels = []
    for ch in range(n_channels):
        coeffs, _ = pywt.cwt(trial_sub[ch], _SCALES, CWT_WAVELET)
        channels.append(np.abs(coeffs))                    # (50, 375)

    cwt_image = np.stack(channels, axis=-1).astype(np.float32)  # (50, 375, 5)
    return cwt_image


def normalise_dataset(X: np.ndarray) -> np.ndarray:
    """
    Per-sample min-max normalisation to [-1, 1].

    Normalization is required for the tanh generator output to be comparable
    with real data during WGAN-GP training.
    """
    mn = X.min(axis=(1, 2, 3), keepdims=True)
    mx = X.max(axis=(1, 2, 3), keepdims=True)
    return (2.0 * (X - mn) / (mx - mn + 1e-8) - 1.0).astype(np.float32)


def preprocess_subject(
    subject: int,
    session: str = "T",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline for one subject/session.

    Parameters
    ----------
    subject : int   1–9
    session : str   'T' (training) or 'E' (evaluation)

    Returns
    -------
    X_cwt : ndarray  (n_trials, 50, 375, 5)   normalised to [-1, 1]
    y     : ndarray  (n_trials,)               0-indexed class labels
    """
    if verbose:
        print(f"  Loading subject {subject} session {session} …", end=" ", flush=True)

    raw  = load_raw(subject, session)
    X, y = extract_epochs(raw)

    cwt_list = [cwt_trial(t) for t in X]
    X_cwt    = np.stack(cwt_list, axis=0)   # (N, 50, 375, 5)
    X_cwt    = normalise_dataset(X_cwt)

    if verbose:
        print(f"{X_cwt.shape[0]} trials  shape={X_cwt.shape[1:]}")

    return X_cwt, y


def split_by_class(X: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
    """Return dict mapping class index → subset of X."""
    return {cls: X[y == cls] for cls in range(N_CLASSES)}
