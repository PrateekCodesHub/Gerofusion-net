"""
process_raw_data.py
===================
Moves raw Kaggle downloads from kaggle_raw/ into the training folder structure:
  dataset/images/{0_Frailty, 1_Sarcopenia, 2_Cognitive_Impairment, 3_Fall_Risk, 4_Depression}
  dataset/tabular/
  dataset/audio/
"""

import os
import shutil
import glob

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR     = os.path.join(BASE_DIR, "dataset", "kaggle_raw")
IMAGES_DIR  = os.path.join(BASE_DIR, "dataset", "images")
TABULAR_DIR = os.path.join(BASE_DIR, "dataset", "tabular")
AUDIO_DIR   = os.path.join(BASE_DIR, "dataset", "audio")

SYNDROME_DIRS = {
    0: os.path.join(IMAGES_DIR, "0_Frailty"),
    1: os.path.join(IMAGES_DIR, "1_Sarcopenia"),
    2: os.path.join(IMAGES_DIR, "2_Cognitive_Impairment"),
    3: os.path.join(IMAGES_DIR, "3_Fall_Risk"),
    4: os.path.join(IMAGES_DIR, "4_Depression"),
}

MAX_PER_CLASS = 800

for d in [IMAGES_DIR, TABULAR_DIR, AUDIO_DIR] + list(SYNDROME_DIRS.values()):
    os.makedirs(d, exist_ok=True)


def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(1 for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))


def copy_images(src_dir, syndrome_idx, prefix):
    dest = SYNDROME_DIRS[syndrome_idx]
    existing = count_images(dest)
    if existing >= MAX_PER_CLASS:
        print(f"  [SKIP] syndrome {syndrome_idx} already has {existing} images")
        return 0
    copied = 0
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if copied + existing >= MAX_PER_CLASS:
                return copied
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                src = os.path.join(root, fname)
                dst = os.path.join(dest, f"{prefix}_{fname}")
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied += 1
    return copied


def main():
    print("\n  Processing raw Kaggle data into training format...\n")

    # 1. Fall Detection → Fall Risk (syndrome 3)
    print("  [1/7] Fall Detection → Fall Risk")
    fall_dir = os.path.join(RAW_DIR, "fall_detection", "fall_dataset", "images")
    if os.path.isdir(fall_dir):
        n = copy_images(fall_dir, 3, "fall")
        print(f"        Copied {n} images")
    else:
        fall_dir2 = os.path.join(RAW_DIR, "fall_detection")
        n = copy_images(fall_dir2, 3, "fall")
        print(f"        Copied {n} images")

    # 2. FER2013 → Depression (sad, fear) + Frailty (neutral, disgust)
    print("  [2/7] FER2013 → Depression + Frailty")
    fer_base = os.path.join(RAW_DIR, "fer2013")
    for fer_class, s_idx, label in [
        ("sad", 4, "Depression"), ("fear", 4, "Depression"),
        ("neutral", 0, "Frailty"), ("disgust", 0, "Frailty"),
    ]:
        for split in ["train", "test"]:
            class_dir = os.path.join(fer_base, split, fer_class)
            if os.path.isdir(class_dir):
                n = copy_images(class_dir, s_idx, f"fer_{fer_class}")
                print(f"        {fer_class}/{split} → {label}: {n} images")

    # 3. Body Performance → Sarcopenia images (no images, but we use FER neutral for Sarcopenia too)
    # Since we don't have dedicated Sarcopenia face images, copy some FER happy/surprise as placeholders
    print("  [3/7] FER2013 (happy/surprise) → Sarcopenia")
    for fer_class in ["happy", "surprise"]:
        for split in ["train", "test"]:
            class_dir = os.path.join(fer_base, split, fer_class)
            if os.path.isdir(class_dir):
                n = copy_images(class_dir, 1, f"fer_{fer_class}")
                print(f"        {fer_class}/{split} → Sarcopenia: {n} images")

    # 4. Cognitive Impairment — use FER angry as visual proxy + rely on OASIS tabular
    print("  [4/7] FER2013 (angry) → Cognitive Impairment")
    for split in ["train", "test"]:
        class_dir = os.path.join(fer_base, split, "angry")
        if os.path.isdir(class_dir):
            n = copy_images(class_dir, 2, "fer_angry")
            print(f"        angry/{split} → Cognitive: {n} images")

    # 5. Tabular CSVs
    print("  [5/7] Copying tabular CSVs")
    tabular_map = {
        "body_fat/bodyfat.csv": "sarcopenia_bodyfat.csv",
        "body_performance/bodyPerformance.csv": "sarcopenia_bodyperformance.csv",
        "cognitive_oasis/oasis_cross-sectional.csv": "cognitive_oasis_cross.csv",
        "cognitive_oasis/oasis_longitudinal.csv": "cognitive_oasis_long.csv",
        "depression_data/b_depressed.csv": "depression_survey.csv",
        "mental_health_survey/survey.csv": "mental_health_survey.csv",
    }
    for src_rel, dst_name in tabular_map.items():
        src = os.path.join(RAW_DIR, src_rel)
        dst = os.path.join(TABULAR_DIR, dst_name)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"        Copied {dst_name}")

    # NHANES CSVs
    nhanes_dir = os.path.join(RAW_DIR, "nhanes")
    if os.path.isdir(nhanes_dir):
        for f in glob.glob(os.path.join(nhanes_dir, "*.csv"))[:4]:
            dst = os.path.join(TABULAR_DIR, f"nhanes_{os.path.basename(f)}")
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
                print(f"        Copied nhanes_{os.path.basename(f)}")

    # 6. Audio files (RAVDESS + speech_emotion)
    print("  [6/7] Audio files → dataset/audio/")
    audio_copied = 0
    for audio_src in ["ravdess_audio", "speech_emotion"]:
        src_dir = os.path.join(RAW_DIR, audio_src)
        if os.path.isdir(src_dir):
            for root, _, files in os.walk(src_dir):
                for fname in files:
                    if fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                        src = os.path.join(root, fname)
                        dst = os.path.join(AUDIO_DIR, f"{audio_src}_{fname}")
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                            audio_copied += 1
    print(f"        Copied {audio_copied} audio files")

    # 7. Summary
    print("\n  ── Summary ──────────────────────────────────")
    for idx in range(5):
        n = count_images(SYNDROME_DIRS[idx])
        name = ["Frailty", "Sarcopenia", "Cognitive_Impairment", "Fall_Risk", "Depression"][idx]
        print(f"  [{idx}] {name:<25} {n:>5} images")

    n_csv = len(glob.glob(os.path.join(TABULAR_DIR, "*.csv")))
    n_aud = sum(1 for f in os.listdir(AUDIO_DIR) if os.path.isfile(os.path.join(AUDIO_DIR, f)))
    print(f"\n  Tabular CSVs : {n_csv}")
    print(f"  Audio files  : {n_aud}")
    print("\n  Done! Now run:  python backend/train_all.py\n")


if __name__ == "__main__":
    main()
