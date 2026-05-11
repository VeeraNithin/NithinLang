# setup.py
from setuptools import setup, find_packages
import os

def read_long_description():
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return (
        "NithinLang V1 — A revolutionary 100% Free, Open-Source, Zero-Cloud, "
        "Multi-Lingual, Ultra-Fast Programming Language."
    )

setup(
    name="nithinlang",
    version="1.0.0",
    author="NithinLang Contributors",
    author_email="nithinlang@opensource.dev",
    description=(
        "NithinLang V1: Free, Open-Source, Zero-Cloud, Multi-Lingual, "
        "Ultra-Fast Programming Language with built-in ML, Game Engine, and AI."
    ),
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/nithinlang/nithinlang",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "nithinlang": [
            "dicts/*.json",
        ],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "llvmlite>=0.41.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pygame-ce>=2.5.0",
        "requests>=2.31.0",
        "rich>=13.0.0",
        "click>=8.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "flake8>=6.0.0",
        ],
        "ai": [
            "ollama>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nithin=nithinlang.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Compilers",
        "Topic :: Software Development :: Interpreters",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)