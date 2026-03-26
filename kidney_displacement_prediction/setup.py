
from setuptools import setup, find_packages

setup(
    name="kidney-displacement-prediction",
    version="1.0.0",
    author="ML Team",
    description="Kidney displacement prediction using ensemble models",
    long_description=open("README.md").read(),
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "api": ["fastapi>=0.104.0", "uvicorn>=0.24.0", "pydantic>=1.10.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.9.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
