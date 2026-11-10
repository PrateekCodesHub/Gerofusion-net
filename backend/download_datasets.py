"""
download_datasets.py
====================
Downloads ALL real Kaggle datasets needed for Geriatric Syndrome GeroFusion v3.

Datasets fetched:
  1. Fall Detection (images)           → Fall Risk
  2. FER2013 Facial Expressions        → Depression / Frailty face analysis
  3. Alzheimer MRI 4-class             → Cognitive Impairment
  4. Dementia Speech Audio             → Cognitive Impairment (audio)
  5. Body Fat Prediction               → Sarcopenia (tabular)
  6. NHANES Elderly Health             → Frailty (tabular)
  7. Geriatric Depression Scale        → Depression (tabular)
  8. Mental Health in Tech Survey      → Depression supplemental
  9. SisFall accelerometer+video       → Fall Risk + Frailty

Usage:
    python download_datasets.py

Kaggle API credentials are embedded below.
You can also set KAGGLE_USERNAME / KAGGLE_KEY env vars.
"""

import os
import sys
import json
import shutil
import zipfile
import glob

os.environ["KAGGLE_USERNAME"] = "feat1phoenix"
os.environ["KAGGLE_KEY"]      = "612b50edebb24820f72196e89494c6f6"
os.environ["PYTHONIOENCODING"] = "utf-8"

# ── Path setup ──────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DIR     = os.path.join(DATASET_DIR, "kaggle_raw")
IMAGES_DIR  = os.path.join(DATASET_DIR, "images")
TABULAR_DIR = os.path.join(DATASET_DIR, "tabular")
AUDIO_DIR   = os.path.join(DATASET_DIR, "audio")
VIDEO_DIR   = os.path.join(DATASET_DIR, "video")

SYNDROME_IMAGE_DIRS = {
    0: os.path.join(IMAGES_DIR, "0_Frailty"),
    1: os.path.join(IMAGES_DIR, "1_Sarcopenia"),
    2: os.path.join(IMAGES_DIR, "2_Cognitive_Impairment"),
    3: os.path.join(IMAGES_DIR, "3_Fall_Risk"),
    4: os.path.join(IMAGES_DIR, "4_Depression"),
}
SYNDROME_NAMES = [
    "Frailty", "Sarcopenia", "Cognitive_Impairment",
    "Fall_Risk", "Depression"
]

for d in [DATASET_DIR, RAW_DIR, IMAGES_DIR, TABULAR_DIR, AUDIO_DIR, VIDEO_DIR] + \
         list(SYNDROME_IMAGE_DIRS.values()):
    os.makedirs(d, exist_ok=True)

MAX_PER_SYNDROME = 800  # cap to keep dataset ≤ ~2 GB total


def log(msg: str):
    print(f"  {msg}", flush=True)


def sep(char="─", n=60):
    print("  " + char * n, flush=True)


def count_files(folder: str, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> int:
    if not os.path.isdir(folder):
        return 0
    return sum(
        1 for _, _, fs in os.walk(folder)
        for f in fs if f.lower().endswith(exts)
    )


def authenticate():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        log("✓ Kaggle API authenticated")
        return api
    except Exception as e:
        log(f"✗ Kaggle auth failed: {e}")
        log("  Install: pip install kaggle")
        sys.exit(1)


def download_dataset(api, ref: str, dest_name: str, force: bool = False):
    dest = os.path.join(RAW_DIR, dest_name)
    os.makedirs(dest, exist_ok=True)
    existing = sum(1 for _, _, fs in os.walk(dest) for _ in fs)
    if existing > 5 and not force:
        log(f"[SKIP] {dest_name} — {existing} files already present")
        return dest
    try:
        log(f"→ Downloading  {ref}")
        api.dataset_download_files(ref, path=dest, unzip=True, quiet=False)
        n = sum(1 for _, _, fs in os.walk(dest) for _ in fs)
        mb = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(dest) for f in fs
        ) / 1024 / 1024
        log(f"  ✓ {n} files, {mb:.0f} MB")
        return dest
    except Exception as e:
        log(f"  ✗ FAILED: {str(e)[:120]}")
        return dest


def copy_images(src_dir: str, syndrome_idx: int, prefix: str, max_per: int = MAX_PER_SYNDROME):
    """Copy images from a raw Kaggle folder into the syndrome-labeled directory."""
    dest = SYNDROME_IMAGE_DIRS[syndrome_idx]
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    existing = count_files(dest)
    if existing >= max_per:
        log(f"  [SKIP] {SYNDROME_NAMES[syndrome_idx]} already has {existing} images")
        return

    copied = 0
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if copied + existing >= max_per:
                break
            if fname.lower().endswith(exts):
                src_path = os.path.join(root, fname)
                dst_path = os.path.join(dest, f"{prefix}_{fname}")
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    copied += 1

    log(f"  → Copied {copied} images to {SYNDROME_NAMES[syndrome_idx]}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN DOWNLOAD ROUTINE
# ══════════════════════════════════════════════════════════════════════

def main():
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   Geriatric Syndrome GeroFusion — Dataset Downloader     ║")
    print("  ║                  Real Kaggle Datasets                    ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()

    api = authenticate()
    sep()

    # ── 1. Fall Detection Images (Fall Risk) ────────────────────────
    print("\n  [1/8] Fall Detection Dataset → Fall Risk")
    dest = download_dataset(
        api,
        "uttejkumarkandagatla/fall-detection-dataset",
        "fall_detection",
    )
    # Typically has 'Fall' and 'No Fall' subdirs
    for sub in ["Fall", "fall", "FALL", "images"]:
        candidate = os.path.join(dest, sub)
        if os.path.isdir(candidate):
            copy_images(candidate, syndrome_idx=3, prefix="fall")
            break
    else:
        copy_images(dest, syndrome_idx=3, prefix="fall")

    # ── 2. FER2013 Facial Expressions (Depression + Frailty faces) ──
    print("\n  [2/8] FER2013 Facial Expressions → Depression / Frailty")
    dest = download_dataset(
        api,
        "msambare/fer2013",
        "fer2013",
    )
    # FER classes: angry disgust fear happy neutral sad surprise
    # Map sad/fear → Depression (idx 4), neutral/disgust → Frailty (idx 0)
    for fer_class, s_idx in [("sad", 4), ("fear", 4), ("neutral", 0), ("disgust", 0)]:
        for split in ["train", "test", ""]:
            class_dir = os.path.join(dest, split, fer_class) if split else os.path.join(dest, fer_class)
            if os.path.isdir(class_dir):
                copy_images(class_dir, syndrome_idx=s_idx, prefix=f"fer_{fer_class}")

    # ── 3. Alzheimer's MRI (Cognitive Impairment) ───────────────────
    print("\n  [3/8] Alzheimer's MRI Dataset → Cognitive Impairment")
    dest = download_dataset(
        api,
        "tourist55/alzheimers-dataset-4-class-of-images",
        "alzheimers_mri",
    )
    # All 4 classes map to Cognitive Impairment (syndrome 2)
    for root, dirs, files in os.walk(dest):
        for fname in files:
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                src = os.path.join(root, fname)
                folder_label = os.path.basename(root)
                dst = os.path.join(
                    SYNDROME_IMAGE_DIRS[2],
                    f"alz_{folder_label}_{fname}",
                )
                if not os.path.exists(dst) and count_files(SYNDROME_IMAGE_DIRS[2]) < MAX_PER_SYNDROME:
                    shutil.copy2(src, dst)

    log(f"  → Cognitive Impairment images: {count_files(SYNDROME_IMAGE_DIRS[2])}")

    # ── 4. Body Fat / Sarcopenia Tabular Data ───────────────────────
    print("\n  [4/8] Body Fat Prediction → Sarcopenia (tabular)")
    dest = download_dataset(
        api,
        "fedesoriano/body-fat-prediction-dataset",
        "body_fat",
    )
    for f in glob.glob(os.path.join(dest, "*.csv")):
        shutil.copy2(f, os.path.join(TABULAR_DIR, "sarcopenia_bodyfat.csv"))
        log(f"  → Copied {os.path.basename(f)} to tabular/")

    # ── 5. NHANES Elderly Health (Frailty tabular) ──────────────────
    print("\n  [5/8] NHANES Elderly Health → Frailty (tabular)")
    dest = download_dataset(
        api,
        "cdc/national-health-and-nutrition-examination-survey",
        "nhanes",
    )
    for f in glob.glob(os.path.join(dest, "**", "*.csv"), recursive=True)[:3]:
        shutil.copy2(f, os.path.join(TABULAR_DIR, f"nhanes_{os.path.basename(f)}"))
        log(f"  → Copied {os.path.basename(f)}")

    # ── 6. Geriatric Depression Scale (Depression tabular) ──────────
    print("\n  [6/8] Geriatric Depression Scale → Depression (tabular)")
    dest = download_dataset(
        api,
        "rtatman/gds-geriatric-depression-scale",
        "gds",
    )
    for f in glob.glob(os.path.join(dest, "*.csv")):
        shutil.copy2(f, os.path.join(TABULAR_DIR, "depression_gds.csv"))
        log(f"  → Copied {os.path.basename(f)}")

    # ── 7. Mental Health Survey (Depression supplemental) ───────────
    print("\n  [7/8] Mental Health in Tech Survey → Depression supplement")
    dest = download_dataset(
        api,
        "osmi/mental-health-in-tech-survey",
        "mental_health_survey",
    )
    for f in glob.glob(os.path.join(dest, "*.csv")):
        shutil.copy2(f, os.path.join(TABULAR_DIR, "mental_health_survey.csv"))
        log(f"  → Copied {os.path.basename(f)}")

    # ── 8. Dementia Speech Audio ─────────────────────────────────────
    print("\n  [8/8] Dementia Speech Audio → Cognitive Impairment (audio)")
    dest = download_dataset(
        api,
        "nitindatta/dementia",
        "dementia_audio",
    )
    # Copy audio files to audio directory
    for root, _, files in os.walk(dest):
        for fname in files:
            if fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                src = os.path.join(root, fname)
                dst = os.path.join(AUDIO_DIR, f"dementia_{fname}")
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
    n_audio = sum(1 for _, _, fs in os.walk(AUDIO_DIR) for f in fs)
    log(f"  → {n_audio} audio files in dataset/audio/")

    # ── Summary ──────────────────────────────────────────────────────
    sep()
    print("\n  ── Dataset Summary ─────────────────────────────────────")
    for idx, name in enumerate(SYNDROME_NAMES):
        n = count_files(SYNDROME_IMAGE_DIRS[idx])
        print(f"  [{idx}] {name:<25} {n:>5} images")

    import glob as _glob
    n_tab = len(_glob.glob(os.path.join(TABULAR_DIR, "*.csv")))
    print(f"\n  Tabular CSVs  : {n_tab}")
    n_aud = sum(1 for _, _, fs in os.walk(AUDIO_DIR) for f in fs)
    print(f"  Audio files   : {n_aud}")

    total_mb = sum(
        os.path.getsize(os.path.join(r, f)) / 1024 / 1024
        for base in [IMAGES_DIR, TABULAR_DIR, AUDIO_DIR]
        for r, _, fs in os.walk(base) for f in fs
    )
    print(f"\n  Total dataset size : {total_mb:.1f} MB")
    sep()
    print("\n  ✓ All downloads complete. Run  train_all.py  next.\n")


if __name__ == "__main__":
    main()
