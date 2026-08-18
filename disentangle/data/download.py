"""
Download datasets from various sources (ENCODE, Synapse, GEO).

Usage:
    python -m data.download --source encode --assay lentiMPRA --cell_type K562 --output_dir data/raw/encode_lentimpra/K562/
    python -m data.download --source synapse --synapse_id syn28469146 --output_dir data/raw/dream/
    python -m data.download --source geo --accession GSE184426 --output_dir data/raw/starrseq/
"""

import requests
import os
import argparse
import subprocess

ENCODE_API = "https://www.encodeproject.org"


def download_encode(assay, cell_type, output_dir):
    """Download MPRA data from ENCODE portal."""
    os.makedirs(output_dir, exist_ok=True)

    url = (
        f"{ENCODE_API}/search/?type=Experiment"
        f"&assay_title={assay}"
        f"&biosample_ontology.term_name={cell_type}"
        f"&format=json"
    )
    response = requests.get(url, headers={"Accept": "application/json"})
    response.raise_for_status()
    experiments = response.json()["@graph"]

    for exp in experiments:
        exp_id = exp["accession"]
        exp_dir = os.path.join(output_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)

        files_url = (
            f"{ENCODE_API}/search/?type=File"
            f"&dataset=/experiments/{exp_id}/"
            f"&format=json&limit=all"
        )
        files_response = requests.get(
            files_url, headers={"Accept": "application/json"}
        )
        files_response.raise_for_status()
        files = files_response.json()["@graph"]

        for f in files:
            if f.get("file_format") in ["tsv", "bed", "bigWig"]:
                file_url = f"{ENCODE_API}{f['href']}"
                filename = f["accession"] + "." + f["file_format"]
                filepath = os.path.join(exp_dir, filename)

                if not os.path.exists(filepath):
                    print(f"Downloading {filename}...")
                    subprocess.run(
                        ["wget", "-q", "-O", filepath, file_url], check=True
                    )

    print(f"Downloaded {len(experiments)} experiments to {output_dir}")


def download_synapse(synapse_id, output_dir):
    """Download from Synapse (requires synapse client and credentials)."""
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(
        ["synapse", "get", "-r", synapse_id, "--downloadLocation", output_dir],
        check=True,
    )


def download_geo(accession, output_dir):
    """Download supplementary files from GEO."""
    os.makedirs(output_dir, exist_ok=True)
    ftp_base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{accession[:7]}nnn/{accession}/suppl/"
    subprocess.run(
        ["wget", "-r", "-np", "-nd", "-P", output_dir, ftp_base],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for DISENTANGLE")
    parser.add_argument(
        "--source", choices=["encode", "synapse", "geo"], required=True
    )
    parser.add_argument("--assay", default=None)
    parser.add_argument("--cell_type", default=None)
    parser.add_argument("--synapse_id", default=None)
    parser.add_argument("--accession", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if args.source == "encode":
        assert args.assay and args.cell_type, "--assay and --cell_type required for ENCODE"
        download_encode(args.assay, args.cell_type, args.output_dir)
    elif args.source == "synapse":
        assert args.synapse_id, "--synapse_id required for Synapse"
        download_synapse(args.synapse_id, args.output_dir)
    elif args.source == "geo":
        assert args.accession, "--accession required for GEO"
        download_geo(args.accession, args.output_dir)
