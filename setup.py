from setuptools import setup, find_packages

setup(
    name="gm_experiment",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "ipykernel>=6.0.0"
    ]
)
