# MNIST Convolutional Neural Network Project

This repository contains a simple Convolutional Neural Network (CNN) model implemented in Python to classify the MNIST dataset. The MNIST dataset is a collection of handwritten digits commonly used for training and testing in the field of machine learning.

## Getting Started

1. **Prerequisites**
   - Python 3.x
   - Necessary libraries (e.g., NumPy, TensorFlow, Keras)

2. **Saving the Data**
   - After training, you can save the model using:
     ```bash
     python mnist_save.py
     ```

3. **Running the Model**
   - To train the model, run:
     ```bash
     python train.py
     ```
   - To run the main script with default settings:
     ```bash
     python main.py
     ```


## Documentation

- **CNN.py**: This file contains the implementation of the CNN model. It defines the architecture and the training process.
- **mnist_dataset.py**: This script handles the loading and preprocessing of the MNIST dataset.
- **train.py**: This script is used to train the model. It includes functions for model initialization, training, and evaluation.
- **main.py**: This is the main entry point of the project. It can be used to run the trained model for predictions and includes a GUI implementation for graphical display and interaction.
- **mnist_save.py**: This script is used to save the MNIST dataset.
- **package.spec**: This file contains specifications for packaging the project, if needed.

## Disclaimer

- **Code Comments**: The code comments are written in Chinese. Users can translate or modify them to better understand and run the code.
- **Interface Setup**: The interfaces are primarily set up for software encapsulation. If encapsulation is not needed, users can adjust them according to their requirements.

## Contributing

Contributions are welcome! Please ensure you follow the existing coding style and add unit tests for any new features.

