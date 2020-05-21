# Machine Learning Introduction Study Notes

A stack of study notes for fundamental machine learning concepts, techniques, models and metrics. Topics covered include (not in order of chapters):

I. **Fundamentals** - supervised learning, unsupervised learning, classification, regression, clustering, 

II. **Linear Regression** - linear regression, regularisation (ridge & lasso), logistic regression, k-nearest neighbours, naïve

III. **Support Vector Machines** - 

IV. **Tree-based techniques** - Bayes, support vector machines, decision trees, regression trees, bagging, pasting, random forests, 
boosting
V. **Other models** - 

VI. **Ensembles** - 

VII. **Performance Evaluation** - 

VIII. **Unsupervised Learning** - PCA, Clustering (k-means clustering, hierarchical clustering)

## References
The following textbooks are used for topical references. The book short-form e.g. `ISTL` will be used as the book title reference in chapter descriptions. If you think the notebooks are insufficient for learning / implementation, please refer to the textbook topics directly.

- An Introduction to Statistical Learning (James, Witten, Hastie, Tibshirani, 2013) --> `ISTL`
- Hands-On Machine Learning with Scikit-Learn (Geron, 2017) --> `HandsML`
- Machine Learning in Action (Harrington, 2012) --> `MLIA`

P.S. `HandsML` has only chapter heading names, not numbers.

## Chapters

### 1. The Machine Learning Landscape
#### 1.1 What is Machine Learning
- `HandsML` --> The Machine Learning Landscape

### 2. End-to-End Machine Learning Project
#### 2.1 - Exploratory Data Analysis (EDA)
#### 2.2 - Feature Engineering
#### 2.3 - Machine Learning
- `HandsML` --> End-to-End Machine Learning Project

### 3. Classification
#### 3.1 - Classification Overview & Performance Measures
- `HandsML` --> MNIST, Training a Binary Classifier, Confusion Matrix, Precision and Recall, Precision/Recall Tradeoff, The ROC Curve
- `ISTL` --> Chap 4.1, Chap 4.2, 4.4.3
#### 3.2 - Validation Set & Cross Validation
- `HandsML` --> Measuring Accuracy Using Cross-Validation
- `ISTL` --> 5.1, 5.2
#### 3.3 - Multiclass Classification & Error Analysis
- `HandsML` --> Multiclass Classification, Error Analysis
- `ISTL` --> 9.4.1, 9.4.2
#### 3.4 - Multilabel Classification
- `HandsML` --> Multilabel Classification
#### 3.5 - Multioutput Classification
- `HandsML` --> Multioutput Classification

### 4. Training Models
#### 4.1a - Linear Regression & Gradient Descent
- `HandsML` --> Linear Regression, Gradient Descent
- `ISTL` --> 3.1.1, 3.2.1
#### 4.1b - Validity of the Model
- `ISTL` --> 3.1.2, 3.1.3, 3.2.2
#### 4.1c - Other Considerations in the Regression Model
- `ISTL` --> 3.3.3
#### 4.2 - Polynomial Regression
- `HandsML` --> Polynomial Regression, Learning Curves
#### 4.3 - Regularised Linear Models
- `HandsML` --> Regularized Linear Models
- `ISTL` --> Chap 6.2.1, 6.2.2, 6.2.3
#### 4.4 - Regularised Linear Models II
- `ISTL` --> Chap 6.2.2
#### 4.4 - Logistic Regression
- `HandsML` --> Logistic Regression
- `ISTL` --> Chap 4.3.1 - 4.3.5
#### 4.5 - k Nearest Neighbours (kNN) Algorithm, Naïve Bayes Algorithm
- `MLIA` --> Chap 2.1 for kNN
- `MLIA` --> Chap 4.1 - 4.5 for Naïve Bayes

### 5. Support Vector Machines
#### 5.1: SVM Classification
- `HandsML` --> Linear SVM Classification
- `ISTL` --> Chap 9.1.1 - 9.1.5, 9.2.1 - 9.2.2
#### 5.2: Non-Linear SVM Classification & SVM Regression
- `HandsML` --> Nonlinear SVM Classification - Polynomial Kernel, Adding Similarity Features, Gaussian RBF Kernel, Computational Complexity, SVM Regression
- `ISTL` --> Chap 9.3.1 - 9.3.2
- also `MLIA` --> Chap 6 (Pending)

### 6. Tree Based Models
#### 6.1: Decision Trees
- `HandsML` --> Training and Visualizing a Decision Tree, Making Predictions, Estimating Class Probabilities, The CART Training Algorithm, Gini Impurity or Entropy?, Instability
- `ISTL` --> Chap 8.1.2, 8.1.3, 8.1.4
#### 6.2: Regression Trees
- `HandsML` --> Regularization Hyperparameters, Regression
- `ISTL` --> Chap 8.1.1
- also `MLIA` --> Chap 3.1, 3.3
- also `MLIA` --> Chap 9.1 - 9.4

### 7. Ensemble Learning
#### 7.1: Voting Classifiers
- `HandsML` --> Voting Classifiers
#### 7.2: Tree-based Ensembles 
- `HandsML` --> Bagging and Pasting, Random Patches and Random Subspaces, Random Forests, Boosting
- `ISTL` --> 8.2.1, 8.2.2
#### 7.3: Boosting
Also has XGB implementation for gradient boosted trees
- `HandsML` --> Boosting
- `ISTL` --> 8.2.3

### 8. Dimensionality Reduction
#### 8.1: Principal Component Analysis
- `HandsML` --> The Curse of Dimensionality, PCA - Preserving the Variance, Principal Components, Projecting Down to d Dimensions, Using Scikit-Learn
- `ISTL` --> 6.3.1, 10.1, 10.2.1, 10.2.2 
#### 8.2: More on Principal Component Analysis
- `HandsML` --> Explained Variance Ratio, Choosing the Right Number of Dimensions, PCA for Compression, Incremental PCA, Randomized PCA
- `ISTL` --> 10.2.3
- also `MLIA` --> 13.1, 13.2

### 9. Clustering Methods
#### 9.1: k-means Clustering
- `ISTL` --> 10.3.1, 10.3.3
#### 9.2: Hierarchical Clustering
- `ISTL` --> 10.3.2
- also `MLIA` --> Chap 10.1, 10.2
