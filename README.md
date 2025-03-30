# Context-Aware Dynamic Integration for Scene Recognition

[Chan Ho Bae](None), [Sangtae Ahn](https://knu-brainai.github.io/)

# Introduction
This repository contains the official implementation of **Context-Aware Dynamic Integration for Scene Recognition**.

## Project Structure

```
scene_recognition/
├── config.py             # Configuration settings
├── train.py              # Training script
├── eval.py               # Evaluation script
├── setup.py              # Project structure initialization
├── prepare_data.py       # Data preparation script
├── requirements.txt      # Required packages
├── models/               # Model related code
├── utils/                # Utility functions
│   ├── dataset.py        # Dataset class
│   └── trainer.py        # Trainer class
├── data/                 # Dataset (auto-generated)
│   └── SUN397_split/
│       ├── train/        # Training data
│       └── val/          # Validation data
└── checkpoints/          # Model storage directory (auto-generated)
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Initialize project structure:
```bash
python setup.py
```

3. Prepare dataset:
```bash
python prepare_data.py
```

## Usage

### 1. Training

```bash
python train.py
```

### 2. Evaluation

```bash
python eval.py
```

## Dataset

The project uses the SUN397 dataset. Data is organized as follows:
- Training images: `data/SUN397_split/train/`
- Training tags: `data/SUN397_split/train_tags.csv`
- Validation images: `data/SUN397_split/val/`
- Validation tags: `data/SUN397_split/val_tags.csv`
