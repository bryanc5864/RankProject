from setuptools import setup, find_packages

setup(
    name="disentangle",
    version="0.1.0",
    description="Learning Biology, Not Noise: Noise-Resistant Sequence-to-Function Modeling",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scipy>=1.12.0",
        "scikit-learn>=1.4.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "h5py>=3.10.0",
        "pysam>=0.22.0",
        "biopython>=1.83",
        "einops>=0.7.0",
        "wandb>=0.16.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "captum>=0.7.0",
        "umap-learn>=0.5.5",
        "pytest>=7.4.0",
    ],
)
