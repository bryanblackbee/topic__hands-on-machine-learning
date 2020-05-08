# Hands on Machine Learning Study Notes

## References
- An Introduction to Statistical Learning (James, Witten, Hastie, Tibshirani, 2013) --> ISTL
- Hands-On Machine Learning with Scikit-Learn (Geron, 2017) --> HandsML
- Machine Learning in Action (Harrington, 2012) --> MLIA
## Chapters
### 1. The Machine Learning Landscape
#### 1.1 What is Machine Learning
- HandsML --> The Machine Learning Landscape

### 2. End-to-End Machine Learning Project
#### 2.1 - Exploratory Data Analysis (EDA)
#### 2.2 - Feature Engineering
#### 2.3 - Machine Learning
- HandsML --> End-to-End Machine Learning Project

### 3. Classification  (Done)
#### 3.1 - Classification Overview & Performance Measures
- HandsML --> MNIST, Training a Binary Classifier, Confusion Matrix, Precision and Recall, Precision/Recall Tradeoff, The ROC Curve
- ISTL --> Chap 4.1, Chap 4.2, 4.4.3
#### 3.2 - Validation Set & Cross Validation
- HandsML --> Measuring Accuracy Using Cross-Validation
- ISTL --> 5.1, 5.2
#### 3.3 - Multiclass Classification & Error Analysis
- HandsML --> Multiclass Classification, Error Analysis
- ISTL --> 9.4.1, 9.4.2
#### 3.4 - Multilabel Classification
- HandsML --> Multilabel Classification
#### 3.5 - Multioutput Classification
- HandsML --> Multioutput Classification

### 4. Training Models
#### 4.1 - Linear Regression & Gradient Descent
- HandsML --> Linear Regression, Gradient Descent
#### 4.2 - Polynomial Regression
- HandsML --> Polynomial Regression, Learning Curves
#### 4.3 - Regularised Linear Models
- HandsML --> Regularized Linear Models
#### 4.4 - Logistic Regression
- HandsML --> Logistic Regression
- ISTL --> Chap 4.3
#### 4.5 - k Nearest Neighbours (kNN) Algorithm, Naïve Bayes Algorithm
- MLIA --> Chap 2.1 for kNN
- MLIA --> Chap 4.1 - 4.5 for Naïve Bayes

### 5. Support Vector Machines (Done)
#### 5.1: SVM Classification
- HandsML --> Linear SVM Classification
- ISTL --> Chap 9.1.1 - 9.1.5, 9.2.1 - 9.2.2
#### 5.2: Non-Linear SVM Classification & SVM Regression
- HandsML --> Nonlinear SVM Classification - Polynomial Kernel, Adding Similarity Features, Gaussian RBF Kernel, Computational Complexity, SVM Regression
- ISTL --> Chap 9.3.1 - 9.3.2
#### Additional Readings
- MLIA --> Chap 6 (Pending)

### 6. Tree Based Models (Done)
#### 6.1: Decision Trees
- HandsML --> Training and Visualizing a Decision Tree, Making Predictions, Estimating Class Probabilities, The CART Training Algorithm, Gini Impurity or Entropy?, Instability
- ISTL --> Chap 8.1.2, 8.1.3, 8.1.4
#### 6.2: Regression Trees
- HandsML --> Regularization Hyperparameters, Regression
- ISTL --> Chap 8.1.1
#### Additional Readings
- MLIA --> Chap 3.1, 3.3
- MLIA --> Chap 9.1 - 9.4, with Model Trees (9.5)

### 7. Ensemble Learning (Done)
#### 7.1: Voting Classifiers
- HandsML --> Voting Classifiers
#### 7.2: Tree-based Ensembles 
- HandsML --> Bagging and Pasting, Random Patches and Random Subspaces, Random Forests, Boosting
- ISTL --> 8.2.1, 8.2.2
#### 7.3: Boosting
Also has XGB implementation for gradient boosted trees
- HandsML --> Boosting
- ISTL --> 8.2.3

### 8. Dimensionality Reduction (Done)
#### 8.1: Principal Component Analysis
- HandsML --> The Curse of Dimensionality, PCA - Preserving the Variance, Principal Components, Projecting Down to d Dimensions, Using Scikit-Learn
- ISTL --> 10.1, 10.2.1, 10.2.2 
#### 8.2: More on Principal Component Analysis
- HandsML --> Explained Variance Ratio, Choosing the Right Number of Dimensions, PCA for Compression, Incremental PCA, Randomized PCA
- ISTL --> 10.2.3

### 9. Clustering Methods (Done)
#### 9.1: k-means Clustering
- ISTL --> 10.3.1, 10.3.3
#### 9.2: Hierarchical Clustering
- ISTL --> 10.3.2
#### Additional Readings
- MLIA --> Chap 10 (with bisecting k-means)
