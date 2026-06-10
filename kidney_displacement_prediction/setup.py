from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).resolve().parent
README = HERE / "README.md"

setup(
    name="kidney-displacement-prediction",
    version="1.0.0",
    author="ML Team",
    description="Kidney displacement prediction using ensemble models",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(where=str(HERE), exclude=["tests", "tests.*"]),
    package_dir={"": str(HERE)},
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "api": ["fastapi>=0.104.0", "uvicorn>=0.24.0", "pydantic>=2.0.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.9.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
