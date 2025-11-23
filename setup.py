#!/usr/bin/env python3
"""
Setup script for HRI30 Action Recognition
Makes the project installable as a Python package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
long_description = (Path(__file__).parent / "README.md").read_text()

# Read requirements
requirements = []
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    # Filter out extra-index-url lines
    requirements = [r for r in requirements if not r.startswith('--')]

setup(
    name="hri30-action-recognition",
    version="1.0.0",
    author="HRI30 Team",
    description="Deep learning framework for industrial human-robot interaction action recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'gpu': [
            'torch>=1.12.0+cu117',
            'torchvision>=0.13.0+cu117'
        ],
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950'
        ]
    },
    entry_points={
        'console_scripts': [
            'hri30-train=main:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="action-recognition, deep-learning, computer-vision, human-robot-interaction",
    project_urls={
        "Documentation": "https://github.com/your-org/hri30-action-recognition",
        "Source": "https://github.com/your-org/hri30-action-recognition",
        "Tracker": "https://github.com/your-org/hri30-action-recognition/issues",
    },
)