# Context-Aware Dynamic Integration for Scene Recognition

This repository contains the implementation of a context-aware dynamic integration model for scene recognition. The model integrates visual features from a CNN backbone with contextual information from text descriptions to improve scene recognition performance.

## Features

- Scene recognition using ResNet backbone networks
- Context-aware dynamic integration of visual and textual features
- Support for both SUN397 and MIT Indoor67 datasets
- Comprehensive data preparation pipeline for easy setup
- Training and evaluation scripts with customizable configurations

## Datasets

This project supports two popular scene recognition datasets:

- **SUN397**: A large-scale scene dataset containing 397 categories with ~100,000 images
- **MIT Indoor67**: A dataset focused on indoor scenes with 67 categories and 15,620 images

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- transformers
- pandas
- tqdm
- numpy

You can install the required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Getting Started

### Data Preparation

The project includes a data preparation script that automatically downloads and prepares the SUN397 and MIT Indoor67 datasets:

```bash
# Prepare both datasets (default)
python prepare_data.py

# Prepare only SUN397 dataset
python prepare_data.py --dataset sun397

# Prepare only MIT Indoor67 dataset
python prepare_data.py --dataset indoor67

# Specify a custom data directory
python prepare_data.py --data_dir /path/to/custom/data/directory
```

### Configuration

Model, training, and dataset configurations can be customized in the `config.py` file. Key options include:

- `DATASET`: Select which dataset to use ('sun397' or 'indoor67')
- `TRAIN_MODE`: Choose training mode ('rgb' for RGB branch only, 'dynamic' for context integration)
- `DEVICE`: Set the device for training and evaluation
- Training hyperparameters (batch size, learning rate, etc.)

### Training

To train the model, run:

```bash
# Train using settings in config.py
python train.py

# To use a specific dataset, modify DATASET in config.py first
```

### Evaluation

Evaluate a trained model using:

```bash
# Evaluate the RGB branch model
python eval.py --model rgb

# Evaluate the dynamic integration model
python eval.py --model dynamic
```

## Model Architecture

The implementation includes two main models:

1. **RGB Branch**: A standard CNN-based model using ResNet (18 or 50) for scene recognition
2. **Dynamic Context Integration**: An enhanced model that combines visual features with textual context using a gate mechanism

## Acknowledgements and Citations

This code builds upon the following open-source projects:

### Semantic-Aware Scene Recognition

The RGB branch implementation and scene recognition architecture were inspired by the work from the VPULab:

```
@article{lopez2020semantic,
  title={Semantic-Aware Scene Recognition},
  author={López-Cifuentes, Alejandro and Escudero-Viñolo, Marcos and Bescós, Jesús and García-Martín, Álvaro},
  journal={Pattern Recognition},
  pages={107256},
  year={2020},
  publisher={Elsevier}
}
```

Source: [Semantic-Aware-Scene-Recognition](https://github.com/vpulab/Semantic-Aware-Scene-Recognition)

### Recognize Anything Model (RAM)

The tag generation model and text embedding components were adapted from the Recognize Anything Model:

```
@article{zhang2023recognize,
  title={Recognize Anything: A Strong Image Tagging Model},
  author={Zhang, Xinyu and Zhang, Huaibo and Zhang, Hulin and Zhou, Yonghong and Yang, Xiao and Li, Baoyuan and Zhu, Lei},
  journal={arXiv preprint arXiv:2306.03514},
  year={2023}
}
```

Source: [Recognize Anything Model](https://github.com/xinyu1205/recognize-anything)

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgements

- SUN397 dataset: [SUN Database](https://vision.princeton.edu/projects/2010/SUN/)
- MIT Indoor67 dataset: [MIT Indoor Scenes](http://web.mit.edu/torralba/www/indoor.html)
