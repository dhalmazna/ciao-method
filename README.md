# CIAO Method - Enterprise Architecture

## Overview
This is a modernized implementation of the CIAO (Contextual Importance Assessment via Obfuscation) method for explainable AI, restructured to follow enterprise-grade architectural patterns with Hydra configuration management and PyTorch Lightning integration.

## Features
- Hydra-based configuration management
- PyTorch Lightning integration
- Factory pattern for components
- Support for medical imaging datasets (prostate/colorectal cancer)
- Modular architecture for easy extensibility

## Structure
```
ciao/
├── components/         # High-level CIAO components
│   ├── segmentation/
│   ├── obfuscation/
│   ├── explainer/
│   └── factory.py
├── modeling/          # Core model implementations
├── data/             # Data loading and processing
├── utils/            # Utility functions
└── scripts/          # Data processing scripts
```

## Installation
```bash
uv sync
```

## Usage
```bash
ciao
# or
python -m ciao
```

## Configuration
The system uses Hydra for configuration management. See `configs/` directory for available configurations.