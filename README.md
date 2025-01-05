Banknote Classification Analysis

Description:
This repository contains the code and analysis results for categorizing genuine and counterfeit banknotes based on features extracted from images. The data is taken from the Data.xlsx file, which is loaded and processed with the pandas library.

Data
The data were extracted from images captured through an industrial camera in grayscale. Feature extraction was performed using the Wavelet Transform method, and includes the following parameters:

Scatter
Asymmetry
Corrugation
Entropy
Procedure
Data separation:

60% of each category is used for training, 20% for validation and 20% for testing.
Normalization of attributes where necessary.
Methods applied:

Dimensional reduction by PCA: Reduction of features in 2 dimensions.
Parzen window classification: test with different sizes of Gaussian windows.
k-Nearest Neighbors (k-NN): test with different values of k.
SVM: Use linear and non-linear SVMs with hyperparameter adjustment.
Comparison of methods based on F1-score.
Optional Analysis:

Modeling of positive samples with a mixed Gaussian distribution.
ROC curve generation for different probability thresholds and numbers of elements in the distribution.

Results
Comparison of classification methods (Parzen, k-NN, SVM).
Evaluation with F1-score for each method.
ROC curve for different probability thresholds.
Requirements
Python 3.8+
Libraries:
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy

