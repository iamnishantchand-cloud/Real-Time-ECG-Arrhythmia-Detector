#!/usr/bin/env python3
"""
ECG Arrhythmia Detector — Data Preparation Pipeline
====================================================
Loads MIT-BIH Arrhythmia Database records from PhysioNet via the `wfdb`
library, segments individual heartbeats around cardiologist-annotated
R-peaks, normalizes amplitude, and splits data by PATIENT (not by beat)
to prevent data leakage.

Dataset: MIT-BIH Arrhythmia Database (physionet.org/content/mitdb)
Each record: ~30 minutes of continuous 2-lead ECG at 360 Hz, with
beat-by-beat annotations by two cardiologists.

Usage:
    python data_prep.py
"""

import os
import json
import time
import numpy as np
import wfdb

# ── Configuration ─────────────────────────────────────────────────────

OUTPUT_DIR = "output"
DEMO_DIR = os.path.join(OUTPUT_DIR, "demo_recordings")

# Beat segmentation window (centered on R-peak)
# A typical heartbeat (PQRST complex) at 360 Hz spans ~250 samples.
# We take 90 samples before the R-peak (captures the P-wave and
# QRS onset) and 160 after (captures the full QRS, ST segment, and T-wave).
PRE_PEAK = 90     # samples before R-peak
POST_PEAK = 160   # samples after R-peak
BEAT_LEN = PRE_PEAK + POST_PEAK  # 250 samples total

# Sampling frequency of MIT-BIH records
FS = 360  # Hz

# PhysioNet database identifier for the wfdb library
PHYSIONET_DB = "mitdb"

# ── AAMI Beat-Class Mapping ──────────────────────────────────────────
# The AAMI (Association for the Advancement of Medical Instrumentation)
# standard groups the 15+ MIT-BIH annotation symbols into 5 superclasses.
# This is the standard grouping used in virtually all published research
# on this dataset.
#
#   N = Normal beats (includes bundle branch blocks & escape beats,
#       which are "normal" in terms of rhythm origin)
#   S = Supraventricular ectopic beats (originate above the ventricles —
#       e.g., atrial premature complexes)
#   V = Ventricular ectopic beats (originate in the ventricles —
#       e.g., PVCs, the most common dangerous arrhythmia)
#   F = Fusion beats (simultaneous normal + ventricular activation,
#       producing a hybrid morphology)
#   Q = Unknown / paced / unclassifiable beats

AAMI_MAPPING = {
    # Normal beats (N)
    "N": "N",   # Normal beat
    "L": "N",   # Left bundle branch block beat
    "R": "N",   # Right bundle branch block beat
    "e": "N",   # Atrial escape beat
    "j": "N",   # Nodal (junctional) escape beat
    # Supraventricular ectopic beats (S)
    "A": "S",   # Atrial premature beat
    "a": "S",   # Aberrated atrial premature beat
    "J": "S",   # Nodal (junctional) premature beat
    "S": "S",   # Supraventricular premature beat
    # Ventricular ectopic beats (V)
    "V": "V",   # Premature ventricular contraction (PVC)
    "E": "V",   # Ventricular escape beat
    # Fusion beats (F)
    "F": "F",   # Fusion of ventricular and normal beat
    # Unknown / Paced (Q)
    "/": "Q",   # Paced beat
    "f": "Q",   # Fusion of paced and normal beat
    "Q": "Q",   # Unclassifiable beat
}

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# ── Patient-Level Train/Test Split ───────────────────────────────────
# We split WHOLE patient records into train vs. test sets.
# This prevents "data leakage" — if we split randomly by beat, the model
# could memorize patient-specific signal characteristics (baseline noise,
# electrode placement artifacts, heart rate) instead of learning general
# arrhythmia morphology patterns.
#
# Records chosen to ensure:
#   - Training set has a diverse mix of normal and abnormal beats
#   - Test set includes patients with known PVCs (207, 208) for compelling demos
#   - Test set includes a mostly-normal patient (100) for contrast

TRAIN_RECORDS = [
    # Use ALL available MIT-BIH records except the 5 test patients.
    # This maximizes training diversity for best generalization.
    # ── Original 10 records ──
    "101",   # Mostly normal beats
    "106",   # Ventricular ectopic beats, fusion beats
    "108",   # Mixed normal and abnormal
    "109",   # PVCs and some atrial premature beats
    "112",   # Mostly normal - good baseline training data
    "115",   # Mostly normal
    "118",   # Normal with some ventricular ectopic
    "119",   # Significant ventricular ectopic activity
    "201",   # Ventricular ectopic beats
    "210",   # PVCs and supraventricular ectopic
    # ── Added in v2 for minority class diversity ──
    "200",   # LBBB with PVCs - V class
    "202",   # Normal with APBs - S class
    "203",   # Heavy ventricular ectopic - V class
    "212",   # RBBB - diverse morphology
    "213",   # VEBs + Fusion beats - CRITICAL for F class
    "214",   # LBBB with PVCs - V class
    "219",   # Normal with some PVCs
    "222",   # Atrial premature beats - S class
    "223",   # Ventricular ectopic - V class
    "228",   # PVCs + APBs - both S and V classes
    "232",   # APBs - CRITICAL for S class (many SVPB)
    "233",   # Heavy PVCs - V class
    # ── Added in v3: ALL remaining MIT-BIH records ──
    "102",   # Normal with PVCs
    "104",   # PVCs, paced beats
    "107",   # Ventricular ectopic
    "111",   # Normal
    "113",   # Normal
    "114",   # Normal with some SVPBs
    "116",   # Normal with VEBs
    "117",   # Normal
    "121",   # Normal
    "122",   # Normal
    "123",   # Normal
    "124",   # Normal with some abnormal
    "205",   # Normal with some PVCs
    "209",   # Normal with PVCs
    "215",   # Normal with PVCs
    "217",   # Paced/PVCs
    "220",   # Normal
    "221",   # PVCs
    "230",   # Normal
    "231",   # Normal with VEBs
    "234",   # Normal
]

TEST_RECORDS = [
    "100",   # Mostly normal + some APBs — good "healthy" demo recording
    "103",   # Normal — clean baseline for testing
    "105",   # Some PVCs — moderate abnormality
    "207",   # Heavy PVC activity — excellent "abnormal" demo recording
    "208",   # Heavy PVC activity — another strong demo recording
]

# Records to save as full continuous signals for live dashboard playback.
# These are carefully chosen test-set patients for the demo:
#   - 100: mostly normal heartbeats (the "healthy patient" demo)
#   - 207: frequent PVCs clearly flagged by model (the "arrhythmia" demo)
#   - 208: another patient with PVCs (backup / variety)
DEMO_RECORDS = ["100", "207", "208"]


# ── Core Functions ────────────────────────────────────────────────────

def load_record(record_id, max_retries=3):
    """
    Load a single MIT-BIH record and its beat annotations from PhysioNet.

    Uses the wfdb library to stream data directly from PhysioNet's servers,
    so no manual download is needed. Each record contains two ECG leads;
    we use only the first (typically MLII - Modified Limb Lead II, the
    standard monitoring lead in clinical settings).

    Includes retry logic with exponential backoff to handle intermittent
    PhysioNet connection drops (common on Windows due to socket limits).

    Returns:
        signal: 1D numpy array of ECG voltage values
        peak_positions: array of sample indices where R-peaks occur
        symbols: list of annotation symbols (one per beat)
    """
    for attempt in range(max_retries):
        try:
            print(f"  Loading record {record_id} from PhysioNet"
                  f"{f' (retry {attempt})' if attempt > 0 else ''}...")
            record = wfdb.rdrecord(record_id, pn_dir=PHYSIONET_DB)
            annotation = wfdb.rdann(record_id, "atr", pn_dir=PHYSIONET_DB)

            # Use only the first ECG lead (MLII in most records)
            signal = record.p_signal[:, 0]

            # Brief pause to avoid overwhelming PhysioNet with rapid requests
            time.sleep(1)

            return signal, annotation.sample, annotation.symbol

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    [!] Attempt {attempt+1} failed: {e}")
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise  # Re-raise on final attempt


def segment_beats(signal, peak_positions, symbols):
    """
    Extract fixed-length windows around each annotated R-peak.

    Each beat is a 250-sample segment: 90 samples before the R-peak
    and 160 samples after. This captures the full PQRST complex:

        P-wave | QRS complex | ST segment | T-wave
              ^--- R-peak is here (the tallest spike)

    Beats whose annotation symbol isn't in the AAMI mapping (e.g.,
    rhythm-change markers like '+' or '~') are skipped, as are beats
    too close to the recording boundaries.

    Returns:
        beats: (N, 250) array of beat waveforms
        labels: (N,) array of integer class labels (0-4)
        valid_positions: (N,) array of R-peak positions for kept beats
    """
    beats = []
    labels = []
    positions = []

    for pos, sym in zip(peak_positions, symbols):
        # Skip annotation types not in our AAMI mapping
        # (e.g., rhythm change markers '+', '~', non-beat annotations)
        if sym not in AAMI_MAPPING:
            continue

        # Skip beats too close to recording start/end to avoid
        # index-out-of-bounds when extracting the fixed window
        if pos - PRE_PEAK < 0 or pos + POST_PEAK > len(signal):
            continue

        # Extract the beat segment centered on the R-peak
        beat = signal[pos - PRE_PEAK : pos + POST_PEAK]
        label = AAMI_MAPPING[sym]

        beats.append(beat)
        labels.append(CLASS_TO_IDX[label])
        positions.append(pos)

    return np.array(beats), np.array(labels), np.array(positions)


def normalize_beats(beats):
    """
    Normalize each beat independently to [0, 1] range.

    Why per-beat normalization?
    - Different patients have different signal amplitudes due to body
      composition, electrode placement, and recording equipment
    - Even within one patient, amplitude can drift over a 30-min recording
    - We want the model to learn SHAPE-based patterns (e.g., the wide
      QRS complex in a PVC) rather than absolute voltage levels

    Each beat is independently scaled so its minimum maps to 0 and
    maximum maps to 1, preserving the waveform morphology.
    """
    normalized = np.zeros_like(beats)
    for i in range(len(beats)):
        beat_min = beats[i].min()
        beat_max = beats[i].max()
        # Guard against division by zero for flat/constant signals
        if beat_max - beat_min > 1e-6:
            normalized[i] = (beats[i] - beat_min) / (beat_max - beat_min)
        else:
            normalized[i] = 0.5  # flat signal → midpoint
    return normalized


def process_records(record_ids, label=""):
    """
    Load, segment, and normalize beats from a list of patient records.

    Prints progress and per-record statistics for transparency.

    Returns:
        all_beats: (total_N, 250) array of normalized beat waveforms
        all_labels: (total_N,) array of integer class labels
    """
    all_beats = []
    all_labels = []

    for rec_id in record_ids:
        try:
            signal, peaks, symbols = load_record(rec_id)
            beats, labels, _ = segment_beats(signal, peaks, symbols)
            beats = normalize_beats(beats)
            all_beats.append(beats)
            all_labels.append(labels)

            # Show class distribution for this record
            unique, counts = np.unique(labels, return_counts=True)
            dist = {CLASS_NAMES[u]: int(c) for u, c in zip(unique, counts)}
            print(f"    > {len(beats)} beats extracted | Classes: {dist}")

        except Exception as e:
            print(f"    [!] Failed to load record {rec_id}: {e}")

    if not all_beats:
        raise RuntimeError(f"No {label} records could be loaded!")

    return np.concatenate(all_beats), np.concatenate(all_labels)


def save_demo_recordings(record_ids):
    """
    Save full continuous recordings for live dashboard playback.

    Unlike the segmented beat data used for training, these files preserve
    the entire continuous signal so the dashboard can animate a scrolling
    ECG waveform. Each .npz file contains:

        signal:        full ECG signal (first lead), raw amplitude
        peaks:         R-peak sample indices (after filtering to AAMI-valid beats)
        labels:        AAMI class index for each beat
        beat_segments: pre-extracted & normalized 250-sample windows
        fs:            sampling frequency (360 Hz)

    Pre-extracting the beat segments here means the dashboard never needs
    to re-segment or re-normalize — it just loads and runs inference.
    """
    os.makedirs(DEMO_DIR, exist_ok=True)

    for rec_id in record_ids:
        try:
            signal, peaks, symbols = load_record(rec_id)
            beats, labels, valid_peaks = segment_beats(signal, peaks, symbols)
            beats_norm = normalize_beats(beats)

            out_path = os.path.join(DEMO_DIR, f"record_{rec_id}.npz")
            np.savez(
                out_path,
                signal=signal,
                peaks=valid_peaks,
                labels=labels,
                beat_segments=beats_norm,
                fs=FS,
            )

            # Show composition for demo quality assurance
            unique, counts = np.unique(labels, return_counts=True)
            dist = {CLASS_NAMES[u]: int(c) for u, c in zip(unique, counts)}
            print(f"    > Demo recording {rec_id} saved | {len(valid_peaks)} beats | {dist}")

        except Exception as e:
            print(f"    [!] Failed to save demo recording {rec_id}: {e}")


# ── Main Pipeline ─────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  ECG Arrhythmia Detector - Data Preparation Pipeline")
    print("=" * 65)

    # ── Step 1: Load ALL unique records once into cache ────────────────
    # Loading each record only once avoids redundant PhysioNet downloads
    # (demo records overlap with test records) and reduces connection issues.
    all_record_ids = list(dict.fromkeys(TRAIN_RECORDS + TEST_RECORDS + DEMO_RECORDS))
    print(f"\n[1/4] Loading {len(all_record_ids)} unique records from PhysioNet...")
    print(f"      Records: {all_record_ids}")

    record_cache = {}  # {record_id: (signal, peaks, symbols)}
    for rec_id in all_record_ids:
        try:
            signal, peaks, symbols = load_record(rec_id)
            record_cache[rec_id] = (signal, peaks, symbols)
        except Exception as e:
            print(f"    [!] Could not load record {rec_id} after retries: {e}")

    print(f"\n  Loaded {len(record_cache)}/{len(all_record_ids)} records successfully")

    # ── Step 2: Process training records ──────────────────────────────
    print(f"\n[2/4] Segmenting TRAINING beats ({len(TRAIN_RECORDS)} patients):")
    train_beats_list, train_labels_list = [], []
    for rec_id in TRAIN_RECORDS:
        if rec_id not in record_cache:
            print(f"    [!] Skipping {rec_id} (not loaded)")
            continue
        signal, peaks, symbols = record_cache[rec_id]
        beats, labels, _ = segment_beats(signal, peaks, symbols)
        beats = normalize_beats(beats)
        train_beats_list.append(beats)
        train_labels_list.append(labels)
        unique, counts = np.unique(labels, return_counts=True)
        dist = {CLASS_NAMES[u]: int(c) for u, c in zip(unique, counts)}
        print(f"    > {rec_id}: {len(beats)} beats | Classes: {dist}")

    X_train = np.concatenate(train_beats_list)
    y_train = np.concatenate(train_labels_list)

    train_unique, train_counts = np.unique(y_train, return_counts=True)
    train_dist = {CLASS_NAMES[u]: int(c) for u, c in zip(train_unique, train_counts)}
    print(f"\n  [OK] Training set total: {X_train.shape[0]} beats")
    print(f"    Class distribution: {train_dist}")

    # ── Step 3: Process test records ─────────────────────────────────
    print(f"\n[3/4] Segmenting TEST beats ({len(TEST_RECORDS)} patients):")
    test_beats_list, test_labels_list = [], []
    for rec_id in TEST_RECORDS:
        if rec_id not in record_cache:
            print(f"    [!] Skipping {rec_id} (not loaded)")
            continue
        signal, peaks, symbols = record_cache[rec_id]
        beats, labels, _ = segment_beats(signal, peaks, symbols)
        beats = normalize_beats(beats)
        test_beats_list.append(beats)
        test_labels_list.append(labels)
        unique, counts = np.unique(labels, return_counts=True)
        dist = {CLASS_NAMES[u]: int(c) for u, c in zip(unique, counts)}
        print(f"    > {rec_id}: {len(beats)} beats | Classes: {dist}")

    X_test = np.concatenate(test_beats_list)
    y_test = np.concatenate(test_labels_list)

    test_unique, test_counts = np.unique(y_test, return_counts=True)
    test_dist = {CLASS_NAMES[u]: int(c) for u, c in zip(test_unique, test_counts)}
    print(f"\n  [OK] Test set total: {X_test.shape[0]} beats")
    print(f"    Class distribution: {test_dist}")

    # ── Step 4: Save demo recordings from cache (no re-download!) ────
    print(f"\n[4/4] Saving DEMO recordings for dashboard playback:")
    os.makedirs(DEMO_DIR, exist_ok=True)
    saved_demos = []
    for rec_id in DEMO_RECORDS:
        if rec_id not in record_cache:
            print(f"    [!] Skipping demo {rec_id} (not loaded)")
            continue
        signal, peaks, symbols = record_cache[rec_id]
        beats, labels, valid_peaks = segment_beats(signal, peaks, symbols)
        beats_norm = normalize_beats(beats)

        out_path = os.path.join(DEMO_DIR, f"record_{rec_id}.npz")
        np.savez(out_path, signal=signal, peaks=valid_peaks,
                 labels=labels, beat_segments=beats_norm, fs=FS)

        unique, counts = np.unique(labels, return_counts=True)
        dist = {CLASS_NAMES[u]: int(c) for u, c in zip(unique, counts)}
        print(f"    > Demo {rec_id} saved | {len(valid_peaks)} beats | {dist}")
        saved_demos.append(rec_id)

    # ── Save processed data to disk ──────────────────────────────────
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    # Save metadata for reproducibility and dashboard consumption
    metadata = {
        "class_names": CLASS_NAMES,
        "class_descriptions": {
            "N": "Normal Beat",
            "S": "Supraventricular Ectopic Beat",
            "V": "Ventricular Ectopic Beat (PVC)",
            "F": "Fusion Beat",
            "Q": "Unknown / Paced Beat",
        },
        "train_records": TRAIN_RECORDS,
        "test_records": TEST_RECORDS,
        "demo_records": saved_demos,
        "beat_length": BEAT_LEN,
        "pre_peak": PRE_PEAK,
        "post_peak": POST_PEAK,
        "sampling_freq": FS,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "train_class_distribution": train_dist,
        "test_class_distribution": test_dist,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  Data preparation complete!")
    print(f"  Training beats : {X_train.shape[0]:>6,}  shape: {X_train.shape}")
    print(f"  Test beats     : {X_test.shape[0]:>6,}  shape: {X_test.shape}")
    print(f"  Demo recordings: {len(saved_demos)} ({saved_demos})")
    print(f"  Output dir     : {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
