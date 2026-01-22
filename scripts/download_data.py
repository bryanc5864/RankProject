#!/usr/bin/env python3
"""
Download data for Rank-Order Learning MPRA project.

Data sources:
1. DREAM-RNN lentiMPRA: https://github.com/trchristensen-99/dream_rnn_lentimpra
2. CAGI5 Saturation Mutagenesis: http://www.genomeinterpretation.org/cagi5-regulation-saturation.html
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CAGI5_DIR = DATA_DIR / "cagi5"


def download_dream_rnn_data():
    """Clone DREAM-RNN repository with lentiMPRA training data."""
    dream_dir = RAW_DIR / "dream_rnn_lentimpra"

    if dream_dir.exists():
        print(f"DREAM-RNN data already exists at {dream_dir}")
        return dream_dir

    print("Cloning DREAM-RNN lentiMPRA repository...")
    subprocess.run(
        [
            "git", "clone",
            "https://github.com/trchristensen-99/dream_rnn_lentimpra.git",
            str(dream_dir)
        ],
        check=True
    )
    print(f"Downloaded to {dream_dir}")
    return dream_dir


def download_cagi5_data():
    """
    Download CAGI5 saturation mutagenesis data.

    Note: CAGI5 data may require manual download or registration.
    This function provides instructions and attempts automated download where possible.
    """
    CAGI5_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("CAGI5 Saturation Mutagenesis Data")
    print("="*60)
    print("""
CAGI5 data sources:

1. Challenge page: http://www.genomeinterpretation.org/cagi5-regulation-saturation.html

2. Associated paper (Nature 2025):
   https://www.nature.com/articles/s44387-025-00053-3

3. Elements covered:
   - 11 disease-associated enhancers: IRF4, IRF6, MYC, SORT1, etc.
   - 10 promoters: TERT, LDLR, F9, HBG1, etc.
   - ~17,500 SNVs and small indels total

Manual download steps:
1. Visit the challenge page above
2. Download the variant effect data for each element
3. Place files in: {cagi5_dir}

Expected file structure:
{cagi5_dir}/
├── enhancers/
│   ├── IRF4_variants.tsv
│   ├── IRF6_variants.tsv
│   └── ...
└── promoters/
    ├── TERT_variants.tsv
    ├── LDLR_variants.tsv
    └── ...
""".format(cagi5_dir=CAGI5_DIR))

    # Create subdirectories
    (CAGI5_DIR / "enhancers").mkdir(exist_ok=True)
    (CAGI5_DIR / "promoters").mkdir(exist_ok=True)

    print(f"Created directory structure at {CAGI5_DIR}")
    return CAGI5_DIR


def main():
    print("="*60)
    print("Rank-Order MPRA Project - Data Download")
    print("="*60)

    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Download DREAM-RNN data
    print("\n[1/2] DREAM-RNN lentiMPRA Data")
    print("-"*40)
    try:
        dream_dir = download_dream_rnn_data()
        print(f"SUCCESS: DREAM-RNN data at {dream_dir}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to clone DREAM-RNN repo: {e}")
        sys.exit(1)

    # CAGI5 data (mostly manual)
    print("\n[2/2] CAGI5 Evaluation Data")
    print("-"*40)
    cagi5_dir = download_cagi5_data()

    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    print(f"""
Next steps:
1. Explore DREAM-RNN data: {RAW_DIR / 'dream_rnn_lentimpra'}
2. Manually download CAGI5 data to: {CAGI5_DIR}
3. Run preprocessing: python scripts/preprocess.py
""")


if __name__ == "__main__":
    main()
