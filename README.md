# Project Overview

This repository provides implementations and experiments for Vision Transformer (ViT) variants and attention mechanisms. The project is structured to separate core models, utility functions, and experimental scripts for easier navigation and reproducibility.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Main Script](#running-the-main-script)
  - [Experiment Scripts](#experiment-scripts)
- [Models](#models)
- [Utilities](#utilities)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Directory Structure

```
/
├── main.py
├── experiments/
│   ├── __init__.py
│   ├── attention.py
│   ├── mhla_pretrained.py
│   ├── sppp.py
│   ├── sppp_mhla_pretrained.py
│   ├── sppp_pretrained.py
│   ├── traditional.py
│   └── traditional_pretrained.py
├── models/
│   ├── __init__.py
│   ├── attention.py
│   ├── mhla.py
│   ├── mhla_models.py
│   ├── sppp.py
│   ├── sppp_mhla.py
│   ├── vit.py
│   └── vit_mhla.py
└── utils/
    ├── __init__.py
    ├── data_utils.py
    └── metrics.py
```

- **main.py**: Entry point for training and evaluation pipelines. Configure experiments and parameters here.
- **experiments/**: Contains scripts defining different experimental setups and training routines for various models and pretrained variants.
- **models/**: Core model definitions and architectures, including Vision Transformer (ViT) variants and specialized attention modules.
- **utils/**: Utility functions for data loading, preprocessing, and performance metrics calculation.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url> && cd <repository-folder>
   ```
2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

> **Note**: Ensure you have Python 3.8+ installed. GPU support requires PyTorch with CUDA.

## Usage

### Running the Main Script

The `main.py` script serves as the unified entry point. You can specify the model, dataset, and other hyperparameters via command-line arguments. For example:
```bash
python main.py --model vit --dataset CIFAR10 --epochs 100 --batch-size 64
```

Use `python main.py --help` to list all available options and configurations.

### Experiment Scripts

Detailed experimental setups are located in the `experiments/` directory. Examples:
- **sppp.py**: Runs experiments using the SPPP (Superpixel Patch Pooling) variant.
- **attention.py**: Tests custom attention mechanisms.
- **traditional.py**: Baseline experiments without specialized modules.

To run an experiment script directly:
```bash
python experiments/sppp.py --config configs/sppp_config.yaml
```

## Models

Model architectures are defined in the `models/` directory:
- **vit.py**: Standard Vision Transformer implementation.
- **mhla.py**: Multi-Head Latent Attention modules and integration.
- **sppp.py**: Superpixel Patch Pooling module and combined ViT variant.

Modify these files to customize model hyperparameters or integrate new architectures.

## Utilities

The `utils/` folder contains helper functions:
- **data_utils.py**: Data loaders, dataset preprocessing, and augmentation utilities.
- **metrics.py**: Evaluation metrics such as accuracy, AUC, and confusion matrix utilities.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Make your changes and commit: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/your-feature-name`.
5. Open a Pull Request describing your changes.

