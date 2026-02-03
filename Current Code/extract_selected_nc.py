from pathlib import Path
import shutil
import pandas as pd

# =================================================
# ROOT SETUP
# =================================================
# ROOT = Everything/
ROOT = Path(__file__).resolve().parents[2]

PROJECT_DIR = ROOT / "GlobalHighPM2.5-source-data"
SOURCE_DIR  = ROOT                      # where .nc files live
OUT_DIR     = PROJECT_DIR / "Data" / "raw_nc"

CSV_FILES = [
    PROJECT_DIR / "train_files.csv",
    PROJECT_DIR / "val_files.csv",
    PROJECT_DIR / "test_files.csv",
]

OUT_DIR.mkdir(parents=True, exist_ok=True)

# =================================================
# COLLECT FILENAMES FROM CSVs
# =================================================
def collect_filenames(csv_paths):
    names = set()
    for csv in csv_paths:
        df = pd.read_csv(csv)
        for p in df["File Path"].astype(str):
            names.add(Path(p).name)
    return names

# =================================================
# COPY SELECTED FILES
# =================================================
def copy_selected_files(source_dir, filenames, out_dir):
    matches = []
    for fname in filenames:
        src = source_dir / fname
        if src.exists():
            matches.append(src)
        else:
            print(f"⚠️  Missing file: {fname}")

    print(f"Requested filenames: {len(filenames)}")
    print(f"Found in source dir: {len(matches)}")

    if not matches:
        raise RuntimeError("No matching .nc files found in source directory!")

    for src in matches:
        dst = out_dir / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)

    print(f"Copied {len(matches)} files to:")
    print(f"  {out_dir}")

# =================================================
# MAIN
# =================================================
if __name__ == "__main__":
    print(f"Source directory: {SOURCE_DIR}")
    filenames = collect_filenames(CSV_FILES)
    print(f"Total unique filenames requested: {len(filenames)}")
    copy_selected_files(SOURCE_DIR, filenames, OUT_DIR)
