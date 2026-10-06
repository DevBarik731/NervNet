# NervNet
This repository contains the source code for a C++ library implementing fundamental machine learning algorithms from scratch, without relying on external ML frameworks.

## Features

Currently implemented:

- Matrix Operations
- Z-score Normalization
- Linear Regression
- Logistic Regression
- Neural Networks
    - Regression
    - Binary Classification
    - Multiclass Classification
- Mini-Batch Gradient Descent
- Adam optimizer

## Performance

This library was used to train a neural network on the **Introvert-Extrovert** dataset and **Mobile Price** dataset from the Kaggle:

- Introvert-Extrovert: https://www.kaggle.com/competitions/playground-series-s5e7/overview
- Mobile Price : https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

### Results

- **96%** accuracy on the DevSet of Introvert-Extrovert dataset.
- **93%** accuracy on the DevSet of Mobile-Price dataset

The previous implementation took **20 minutes** to train on the Introvert-Extrovert dataset. After using mini-batches, the training time decreased to **2–3 minutes** while maintaining the same accuracy.

It further decreased to **30-60 seconds** after implementing Adam.

## Future Plans

I plan to add more machine learning algorithms and utilities as I continue learning the theoretical foundations of machine learning.