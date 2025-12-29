# American Sign Language CNN Classifier
## Overview
This project implements a Convolutional Neural Network (CNN) to recognize American Sign Language (ASL) hand signs from image data. The model learns visual patterns in hand gestures and classifies them into their corresponding ASL letters, demonstrating the application of deep learning to computer vision tasks.

## Files in This Repository
* `ASL-CNN.ipynb`
  Jupyter Notebook containing data loading, preprocessing, CNN architecture design, model training, and evaluation.
* `sign_mnist_train.csv`
  Training dataset containing labeled grayscale images of American Sign Language hand signs.
* `sign_mnist_test.csv`
  Test dataset used to evaluate the trained model on unseen ASL hand sign images.

## Dataset
The dataset consists of grayscale images of American Sign Language hand signs formatted in a MNIST like structure. Each row represents a flattened image with an associated label indicating the corresponding ASL letter. Separate training and test files are used to train the model and evaluate its performance.

## Methodology
1. Load and preprocess image data
2. Normalize and reshape images for CNN input
3. Build a convolutional neural network architecture
4. Train the model on labeled ASL image data
5. Evaluate performance on validation or test data

## Model
* Architecture: Convolutional Neural Network (CNN)
* Task: Multi class image classification
* Input: Hand sign images
* Output: Predicted ASL letter class

## Results
Model performance is evaluated within the notebook using accuracy and loss metrics. The results show that the CNN is capable of learning meaningful representations for ASL hand sign classification.

## How to Run
1. Clone the repository
2. Install required Python dependencies
3. Open the Jupyter Notebook
4. Run all cells to train and evaluate the model

## Dependencies
* Python3
* NumPy
* Pandas
* Matplotlib
* TensorFlow
* Jupyter Notebook
