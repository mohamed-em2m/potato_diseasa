# Potato Disease Leaf Classification Project

## Overview

This project focuses on classifying potato leaf diseases using deep learning models. The dataset used contains images categorized into three classes: Late Blight, Early Blight, and Healthy. The models are built using PyTorch and TensorFlow/Keras, and the project includes data preprocessing, model training, evaluation, and visualization of results.

## Dataset

The dataset is organized into three splits:

- **Training**: Images used for training the model.
  - Late Blight: 1132 images
  - Early Blight: 1303 images
  - Healthy: 816 images

- **Validation**: Images used for validating the model.
  - Late Blight: 151 images
  - Early Blight: 163 images
  - Healthy: 102 images

- **Testing**: Images used for testing the final model performance.
  - Late Blight: 141 images
  - Early Blight: 162 images
  - Healthy: 102 images

The dataset directory structure is as follows:
Training/
        Late_Blight/
        Early_Blight/
        Healthy/
    Validation/
        Late_Blight/
        Early_Blight/
        Healthy/
    Testing/
        Late_Blight/
        Early_Blight/
        Healthy/
## Preprocessing

The following preprocessing steps were applied to the images:

- Random resized cropping
- Random horizontal flipping
- Random rotation
- TrivialAugmentWide augmentation
- Conversion to tensor format

## Model Architecture

Two Convolutional Neural Network (CNN) models were implemented:

### CNNModel1

- **Layers**:
  - Batch Normalization
  - Convolutional Layers with ReLU activations
  - Max Pooling
  - Fully Connected Layers

### CNNModel2

- **Layers**:
  - Batch Normalization
  - Convolutional Layers with ReLU activations
  - Max Pooling
  - Fully Connected Layers

## Training

Models were trained over 100 epochs with the Adam optimizer. Learning rates were adjusted using a ReduceLROnPlateau scheduler based on validation loss.

## Results

### CNNModel1

- **Final Training Accuracy**: ~91%
- **Final Validation Accuracy**: ~91%

### CNNModel2

- **Final Training Accuracy**: ~97%
- **Final Validation Accuracy**: ~97%
- **Test Accuracy**: 98.44%

Classification Report for CNNModel2:

## Visualization

Plots of training and validation loss and accuracy over epochs are included to visualize the model's progress.

## Dependencies

- Python 3.10
- NumPy
- Pandas
- Matplotlib
- Seaborn
- TensorFlow
- PyTorch
- Scikit-learn
- XGBoost
- Torchvision

## Instructions

1. Clone this repository.
2. Install the necessary dependencies.
3. Run the Jupyter Notebook or Python script to train the models and evaluate their performance.

## Acknowledgments

Thanks to Kaggle and the contributors of the datasets and libraries used in this project.
