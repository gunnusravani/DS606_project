#!/usr/bin/env python3
"""Setup script for DS606 project."""

from setuptools import setup, find_packages

setup(
    name="ds606",
    version="0.0.1",
    description="DS606: Cross-lingual safety alignment transfer in LLMs",
    author="Sravani Gunnu, Aryan Kashyap, Shahab Ahmad",
    author_email="sravani.gunnu@iitb.ac.in",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "trl>=0.6.0",
        "datasets>=2.10.0",
        "pandas>=1.5.0",
        "pyyaml>=6.0",
        "tqdm>=4.60.0",
        "bitsandbytes>=0.35.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort", "flake8"],
        "notebooks": ["jupyter", "notebook", "ipython"],
        "translation": ["googletrans>=4.0.0rc1"],
    },
    entry_points={
        "console_scripts": [
            "ds606=ds606.cli:main",
        ],
    },
)
