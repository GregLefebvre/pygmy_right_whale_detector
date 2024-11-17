# pygmy_right_whale_detector

![me](https://github.com/GregLefebvre/pygmy_right_whale_detector/blob/main/output.gif)

## What the project does
This project provides multiple methods to detect PRW (a species of whale) from underwater recordings. It includes:
- A CNN-based approach (built on TensorFlow).
- A clustering approach using encoded data from a custom-built encoder.
- Techniques such as Dropout, L2 regularization, exponential learning rate decay, data augmentation, and handling multiple false positive classes.

**Note**: Data is not provided and is not free to use.

## Why the project is useful
This repository demonstrates the implementation of convolutional neural networks (CNNs) for detecting biological sounds, specifically PRW calls, using their spectrograms.

## How to get started
1. Install TensorFlow and all the required Python libraries.
2. Prepare a large dataset for training the model.
3. Use `homemade_neural_network.py` to train the model on your dataset.
4. Use `predict_neural_network.py` to make predictions on a new dataset.

## Where to get help
If you have questions or need support, send a private message here.

## Maintainers
This project is maintained by **Grégoire Lefebvre** and **Malo Quimbert**. It was started in September 2024.
