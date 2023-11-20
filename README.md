# Computer Vision Models Comparison

This repository contains code for implementing various Computer Vision models using PyTorch and evaluating their performance on the MNIST dataset. The objective of this lab is to explore different architectures such as CNN, Faster R-CNN, VGG16, AlexNet, and Vision Transformer (ViT) for image classification tasks.

## Lab Objectives

### Part 1: CNN Classifier and Faster R-CNN

1. **CNN Classifier**:
   - Implemented a CNN architecture using PyTorch to classify the MNIST dataset.
   - Defined layers including Convolution, Pooling, and Fully Connected layers.
   - Tuned hyperparameters such as Kernels, Padding, Stride, Optimizers, Regularization, etc.
   - Ran the model in GPU mode for faster computation.

2. **Faster R-CNN**:
   - Implemented Faster R-CNN architecture for object detection on the MNIST dataset.
   - Compared the CNN Classifier and Faster R-CNN using metrics like Accuracy, F1 Score, Loss, and Training Time.

3. **Fine-tuning VGG16 and AlexNet**:
   - Fine-tuned pre-trained models (VGG16 and AlexNet) on the MNIST dataset.
   - Evaluated and compared the performance of the fine-tuned models with CNN and Faster R-CNN.

### Part 2: Vision Transformer (ViT)

1. **Vision Transformer (ViT)**:
   - Established a Vision Transformer architecture from scratch following the provided tutorial.
   - Performed image classification tasks on the MNIST dataset using ViT.

2. **Model Comparison**:
   - Analyzed and interpreted the results obtained from ViT and compared them with the models from Part 1.

## Usage

### Setup and Requirements
- Use Google Colab or Kaggle for running the Jupyter notebooks.
- PyTorch library for building and training neural networks.
- MNIST dataset available at [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

### Instructions
1. Run the respective Jupyter notebooks for each part of the lab.
2. Follow the code and comments for model implementation and evaluation.
3. Compare model performances based on the provided metrics.

### Comparison Results:

#### CNN:
- **Accuracy**: 98.5%
- **F1 Score**: 0.984
- **Loss**: 0.048
- **Training Time**: 10 minutes

#### Faster R-CNN:
- **Accuracy**: 97.8%
- **F1 Score**: 0.976
- **Loss**: 0.065
- **Training Time**: 20 minutes

#### VGG16 (Fine-tuned):
- **Accuracy**: 98.9%
- **F1 Score**: 0.988
- **Loss**: 0.042
- **Training Time**: 25 minutes

#### AlexNet (Fine-tuned):
- **Accuracy**: 97.2%
- **F1 Score**: 0.970
- **Loss**: 0.075
- **Training Time**: 18 minutes

#### ViT:
- **Accuracy**: 99.1%
- **F1 Score**: 0.992
- **Loss**: 0.035
- **Training Time**: 30 minutes

### Conclusion

- ViT outperforms other models with the highest accuracy and F1 score.
- Fine-tuned VGG16 follows closely in performance metrics.
- Training time varies among models, with ViT taking the longest and CNN the shortest.

## Summary

This lab provided hands-on experience in building various neural architectures for Computer Vision tasks. Experimenting with different models allowed for a deeper understanding of their performance, strengths, and limitations.
