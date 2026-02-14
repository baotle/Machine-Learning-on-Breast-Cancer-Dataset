# Machine-Learning-on-Breast-Cancer-Dataset
This repository and its related code were used to train a PyTorch model capable of classifying benign and malignant breast cancer. The dataset is based on the open dataset "Breast Cancer Wisconsin (Diagnostic)". This project has two aims:

1) To train a model that can achieve high classification test accuracy (>95%) for this dataset.
2) For the author to practice and demonstrate his ML/AI skills with PyTorch. 

In the beginning two standard models were employed: a Fully-Connected Neural Network (FCN) with 3 layers and a 1-D Convolutional Neural Network (CNN) with 2 Convolutional Layers and a single Fully-Connected Layer.
Out of scientific curiosity, the rows of the tabular data were transformed into a matrix, the idea being twofold. Firstly, a matrix may allow the neural network to find more elaborate representations of innate features, especially with a CNN. Secondly, those matrices could then be used to train a rich tapestry of established 2-D models already established in other tasks. ( CNN, ResNet, Transformers, etc. )

# Discussion
While the primary aim of this project was achieved with train and test accuracies of around 96.5/99.5 for the best-performing model, a more streamlined approach using SVM, Decision Trees, and the sklearn library yielded similar results. ( https://www.kaggle.com/code/mragpavank/breast-cancer-wisconsin ). Strongly implying that not every classification problem needs an elaborate PyTorch Model to be solved. In fact, the best performing metrics to traintime ratio were achieved by variants of the Fully-Connected Network. Similar results were achieved by the C1NN at (96.2 / 99.76 ). Models trying to train on the transformed dataset, completely flatlined or only gaining negligible amounts of performance gain after 5000 epochs. This indicates that 

A) 
B)
However, within this project, the author learned many concepts, coding routines, and deeper insights into implementing models in PyTorch, resulting in the completion of aim 2.

# Looking Forward


# Ressources
Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.
