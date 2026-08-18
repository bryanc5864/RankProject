#!/usr/bin/env python3
"""
Download data for Rank-Order Learning MPRA project.

Data sources:
1. DREAM-RNN lentiMPRA code: https://github.com/trchristensen-99/dream_rnn_lentimpra
   (the HDF5 activity files themselves live on Zenodo 10.5281/zenodo.14145285)
2. Prix Fixe / de-Boer DREAM challenge code:
   https://github.com/de-Boer-Lab/random-promoter-dream-challenge-2022
   (the human MPRA tables live on Zenodo 10.5281/zenodo.10633252 as human_mpra_data.tar.gz)
3. CAGI5 Saturation Mutagenesis: http://www.genomeinterpretation.org/cagi5-regulation-saturation.html
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


def clone_deboer_prixfixe():
    """Clone the de-Boer DREAM challenge repo (supplies the Prix Fixe modules)."""
    deboer_dir = RAW_DIR / "deboer_dream"

    if deboer_dir.exists():
        print(f"Prix Fixe code already exists at {deboer_dir}")
        return deboer_dir

    print("Cloning de-Boer random-promoter-dream-challenge-2022 repository...")
    subprocess.run(
        [
            "git", "clone",
            "https://github.com/de-Boer-Lab/random-promoter-dream-challenge-2022.git",
            str(deboer_dir)
        ],
        check=True
    )
    print(f"Downloaded to {deboer_dir}")
    return deboer_dir


def print_zenodo_instructions():
    """The bulk data is on Zenodo, not in either git repo."""
    print()
    print("Zenodo downloads (neither repo ships the data itself)")
    print(f"""
1. Human MPRA training tables (K562/HepG2/WTC11_clean.tsv, ~29 MB)
   Zenodo 10.5281/zenodo.10633252

     curl -L 'https://zenodo.org/records/10633252/files/human_mpra_data.tar.gz?download=1' \\
         -o human_mpra_data.tar.gz
     tar -xzf human_mpra_data.tar.gz -C {RAW_DIR / 'deboer_dream'}

   Gives {RAW_DIR / 'deboer_dream' / 'human_mpra'}/K562_clean.tsv (226,253 rows,
   columns seq_id/seq/mean_value/fold). The 10-fold split is the published `fold`
   column, so prepare_fold_data() reproduces the splits exactly.

2. lentiMPRA activity + aleatoric HDF5 files (~1.8 GB tarball)
   Zenodo 10.5281/zenodo.14145285

     curl -L 'https://zenodo.org/record/14145285/files/data.tar.gz?download=1' -o data.tar.gz
     tar -xzf data.tar.gz -C {RAW_DIR / 'dream_rnn_lentimpra'}

   Gives lentiMPRA_K562_activity_and_aleatoric_data.h5 and the HepG2 counterpart.
""")


def download_cagi5_data():
    """
    Download CAGI5 saturation mutagenesis data.

    Note: CAGI5 data may require manual download or registration.
    This function provides instructions and attempts automated download where possible.
    """
    CAGI5_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("CAGI5 Saturation Mutagenesis Data")
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

    # create subdirectories
    (CAGI5_DIR / "enhancers").mkdir(exist_ok=True)
    (CAGI5_DIR / "promoters").mkdir(exist_ok=True)

    print(f"Created directory structure at {CAGI5_DIR}")
    return CAGI5_DIR


def main():
    print("Rank-Order MPRA Project - Data Download")

    # ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # download DREAM-RNN code
    print("\n[1/4] DREAM-RNN lentiMPRA repo")
    try:
        dream_dir = download_dream_rnn_data()
        print(f"SUCCESS: DREAM-RNN code at {dream_dir}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to clone DREAM-RNN repo: {e}")
        sys.exit(1)

    # Prix Fixe architecture modules
    print("\n[2/4] Prix Fixe / de-Boer DREAM challenge repo")
    try:
        deboer_dir = clone_deboer_prixfixe()
        print(f"SUCCESS: Prix Fixe code at {deboer_dir}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to clone de-Boer repo: {e}")
        sys.exit(1)

    # bulk data lives on Zenodo
    print("\n[3/4] Zenodo data downloads")
    print_zenodo_instructions()

    # CAGI5 data (mostly manual)
    print("\n[4/4] CAGI5 Evaluation Data")
    cagi5_dir = download_cagi5_data()

    print()
    print("Download complete!")
    print(f"""
Next steps:
1. Run the two Zenodo commands printed above.
2. Manually download CAGI5 data to: {RAW_DIR / 'dream_rnn_lentimpra' / 'data' / 'CAGI5'}
   (the training/eval scripts read challenge_*.tsv from there, not from {CAGI5_DIR})
3. Train: python scripts/train_deboer_rankloss.py --loss_type mse --n_test_folds 1
""")


if __name__ == "__main__":
    main()
