# MNIST Neural Network

A comprehensive neural network implementation for handwritten digit recognition using the MNIST dataset. This project includes both a Jupyter notebook for training the model and an interactive GUI application for real-time digit prediction.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the GUI Application](#running-the-gui-application)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This project implements a feed-forward neural network using TensorFlow/Keras to classify handwritten digits (0-9) from the MNIST dataset. The implementation includes:

1. **Training Pipeline**: Jupyter notebook with comprehensive data analysis, model training, and evaluation
2. **Interactive GUI**: Tkinter-based drawing canvas for real-time digit prediction
3. **Pre-trained Model**: Ready-to-use model weights for immediate testing

## Features

- **Interactive Drawing Canvas**: Draw digits with your mouse and get instant predictions
- **Real-time Prediction**: Live confidence scores for all digit classes
- **Model Visualization**: Complete training process with performance metrics
- **High Accuracy**: Achieves ~98% accuracy on the test dataset
- **CUDA Support**: Optimized for GPU acceleration (WSL Ubuntu CUDA compatible)

## Project Structure

```
mnist-NN/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── model.ipynb                  # Jupyter notebook for training
├── canvas.py                    # GUI application for digit prediction
├── mnist_model.weights.h5       # Trained model weights
├── mnist_complete_model.h5      # Complete saved model
└── .gitignore                   # Git ignore file
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pillow (PIL)
- Tkinter (usually included with Python)
- Jupyter (for running the notebook)
- CUDA-compatible GPU (optional, for acceleration)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Johan-Artman/mnist-NN.git
   cd mnist-NN
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **For GUI support** (Linux users):
   ```bash
   sudo apt-get install python3-tk
   ```

## Usage

### Training the Model

1. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook model.ipynb
   ```

2. **Run all cells** to:
   - Load and preprocess the MNIST dataset
   - Define the neural network architecture
   - Train the model with visualization
   - Evaluate performance with detailed metrics
   - Save the trained model weights

3. **Access the MNIST dataset**: The notebook automatically downloads the dataset, or you can get it from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

### Running the GUI Application

1. **Ensure model weights exist**: The `mnist_model.weights.h5` file should be present (generated from the notebook).

2. **Launch the application**:
   ```bash
   python canvas.py
   ```

3. **How to use**:
   - Draw a digit on the white canvas using your mouse
   - Click "Predict" to get the model's prediction
   - View the predicted digit and confidence score
   - Click "Clear" to start over

## Model Architecture

The neural network uses a simple but effective feed-forward architecture:

```
Input Layer:    784 neurons (28×28 flattened image)
Hidden Layer 1: 256 neurons (ReLU activation)
Hidden Layer 2: 128 neurons (ReLU activation)  
Hidden Layer 3: 64 neurons (ReLU activation)
Output Layer:   10 neurons (Linear activation + Softmax)
```

**Training Configuration**:
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: Default (typically 32)
- **Epochs**: 40 (configurable in notebook)

## Performance

The model achieves excellent performance on the MNIST test set:

- **Overall Accuracy**: ~98.2%
- **Training Time**: ~5-10 minutes (with CUDA GPU)
- **Model Size**: ~2.9 MB
- **Per-digit Accuracy**: 97-99% across all digits (0-9)

## Technical Details

- **Environment**: Developed and tested on WSL Ubuntu with CUDA support
- **Framework**: TensorFlow 2.x with Keras API
- **Image Processing**: PIL for GUI canvas image handling
- **Data Preprocessing**: Normalization to [0,1] range, flattening to 784-dimensional vectors
- **GUI Framework**: Tkinter for cross-platform compatibility


**Note**: This project was developed and tested on WSL Ubuntu with CUDA support, but should work on any system with the required dependencies installed.
