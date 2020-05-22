# Machine Learning Introduction Study Notes
A stack of study notes for fundamental machine learning concepts, techniques, models and metrics.
## References
The following textbooks are used for topical references. The book short-form e.g. `ISTL` will be used as the book title reference in chapter descriptions. If you think the notebooks are insufficient for learning / implementation, please refer to the textbook topics directly.

- An Introduction to Statistical Learning (James, Witten, Hastie, Tibshirani, 2013) --> `ISTL`
- Hands-On Machine Learning with Scikit-Learn (Geron, 2017) --> `HandsML`
- Machine Learning in Action (Harrington, 2012) --> `MLIA`

P.S. `HandsML` has only chapter heading names, not numbers.

## Chapters

### 1. The Machine Learning Landscape
1.1 gives an overall workflow of modelling using different models using the GDP and happiness dataset.

- `HandsML`
    - The Machine Learning Landscape

#### 1.1 What is Machine Learning
### 2. End-to-End Machine Learning Project
2.1, 2.2 and 2.3 cover different stages of the ML project, including EDA, feature engineering and machine learning.

- `HandsML`
    - End-to-End Machine Learning Project
#### 2.1 - Exploratory Data Analysis (EDA)
#### 2.2 - Feature Engineering
#### 2.3 - Machine Learning
### 3. Classification
3.1 introduces the goals classification problem and contrasts it with the regression problem. It then demonstrates using a **binary classifier** to classify numbers in the MNIST dataset. With the results, it introduces performance metrics for classification problems including **accuracy**, **confusion matrix**, **recall**, **specificity**, **precision**, **f1-score**, **ROC curve** and **ROC-AUC** values.

3.2 discusses resampling methods to handle lack of volume of data and to measure the stability of a model. It introduces the **train-test split**, then extends this method to **cross-validation** and **bootstrapping**.

3.3 extends the binary classifier to the **multiclass classifier** problem. In particular using the **one-vs-all** strategy or the **one-vs-one** strategy. 

3.4 and 3.5 implements **multi-label classification** and **multi-output classification** respectively. Multi-label classification is when a classifier can simultaneously predict two (or more) results from two groups of categories. Multi-output classification is where each result can be one of more than 2 possibilities.

#### 3.1 - Classification Overview & Performance Measures
- `HandsML` 
    - MNIST
    - Training a Binary Classifier
    - Confusion Matrix
    - Precision and Recall
    - Precision/Recall Tradeoff,
    - The ROC Curve
- `ISTL`
    - Chap 4.1, 4.2, 4.4.3
- `MLIA`
    - Chap 7.7
#### 3.2 - Validation Set & Cross Validation
- `HandsML`
    - Measuring Accuracy Using Cross-Validation
- `ISTL`
    - Chap 5.1, 5.2
#### 3.3 - Multiclass Classification & Error Analysis
- `HandsML`
    - Multiclass Classification, Error Analysis
- `ISTL`
    - Chap 9.4.1, 9.4.2
#### 3.4 - Multilabel Classification
- `HandsML`
    - Multilabel Classification
#### 3.5 - Multioutput Classification
- `HandsML`
    - Multioutput Classification

### 4. Training Models
4.1 introduces **Linear Regression** and optimisation using the **normal equations** and **gradient descent**. It then discusses the validity of the estimates using concepts like **bias**, **variance**, **standard errors**, **confidence intervals** and **hypothesis testing**.

4.2 covers extends linear regression to **polynomial regression** and model selection using **learning curves**.

4.3 discusses regularisation of the model using **ridge regression** and **lasso regression**. It also introduces a combined constraint technique called **elastic nets**.

4.4 extends linear regression to **logistic regression**. It also considers the multiclass problem and introduces **softmax regression**.

4.5 introduces different learning models for classification, including **k-nearest neighbours** and **naïve Bayes classification**.

#### 4.1a - Linear Regression & Gradient Descent
- `HandsML`
    - Linear Regression
    - Gradient Descent
- `ISTL`
    - Chap 3.1.1, 3.2.1
#### 4.1b - Validity of the Model
- `ISTL`
    - Chap 3.1.2, 3.1.3, 3.2.2
#### 4.1c - Other Considerations in the Regression Model
- `ISTL`
    - Chap 3.3.3
#### 4.2 - Polynomial Regression
- `HandsML`
    - Polynomial Regression
    - Learning Curves
- `ISTL`
    - Chap 7.1
#### 4.3 - Regularised Linear Models
- `HandsML`
    - Regularized Linear Models
- `ISTL`
    - Chap 6.2.1, 6.2.2, 6.2.3
#### 4.3 - Regularised Linear Models II
- `ISTL`
    - Chap 6.2.2
#### 4.4 - Logistic Regression
- `HandsML`
    - Logistic Regression
- `ISTL`
    - Chap 4.3.1 - 4.3.5
#### 4.5 - k Nearest Neighbours (kNN) Algorithm, Naïve Bayes Algorithm
- kNN: `MLIA`
    - Chap 2.1
- Naïve Bayes: `MLIA`
    - Chap 4.1 - 4.5

### 5. Support Vector Machines
5.1 covers briefly the **Hyperplane**, then introduces the **Maximum Margin Classifier** or hard margin classification. It then extends the discussion to the **Support Vector Classifier** or soft margin classification that allows for violations.

5.2 covers the **Support Vector Machine**, using polynomial kernels and radial kernels to allow for nonlinear boundaries. It then briefly covers **Support Vector Regression**.
#### 5.1: SVM Classification
- `HandsML`
    - Linear SVM Classification
- `ISTL`
    - Chap 9.1.1 - 9.1.5 
    - Chap 9.2.1 - 9.2.2
#### 5.2: Non-Linear SVM Classification & SVM Regression
- `HandsML`
    -  Nonlinear SVM Classification - Polynomial Kernel
    -  Adding Similarity Features
    -  Gaussian RBF Kernel
    -  Computational Complexity
    -  SVM Regression
- `ISTL`
    -  Chap 9.3.1 - 9.3.2

Additional Readings
- `MLIA`
    -  Chap 6.1, 6.2, 6.5, 6.7
### 6. Tree Based Models
6.1 covers how to read a **classification tree**, then explains how to train a decision tree by first splitting a sample into nodes and then calculating node purity using either the **Gini index** or **Cross-Entropy**. 

6.2 covers **regression trees**, an extension of classification trees in 6.1. It then discusses how to regularise the model using different pruning methods.
#### 6.1: Decision Trees
- `HandsML`
    - Training and Visualizing a Decision Tree
    - Making Predictions
    - Estimating Class Probabilities
    - The CART Training Algorithm
    - Gini Impurity or Entropy?
    - Instability
- `ISTL`
    - Chap 8.1.2, 8.1.3, 8.1.4
#### 6.2: Regression Trees
- `HandsML` --> Regularization Hyperparameters, Regression
- `ISTL` --> Chap 8.1.1

Additional Readings
- `MLIA`
    - Chap 3.1, 3.3
    - Chap 9.1 - 9.4
### 7. Ensemble Learning
7.1 introduces ensemble learning with **voting classifiers** and how they can be used to improve a single model. There are two flavours of voting - hard voting or soft voting.

7.2 discusses tree-based ensembles including bootstrap aggregating or **bagging**, **pasting**, **random forests** and **extra-trees**.

7.3 discusses **boosting**.
#### 7.1: Voting Classifiers
- `HandsML`
    - Voting Classifiers
#### 7.2: Tree-based Ensembles 
- `HandsML`
    - Bagging and Pasting
    - Random Patches and Random Subspaces
    - Random Forests
- `ISTL` 
    - Chap 8.2.1, 8.2.2
#### 7.3: Boosting
Also has XGB implementation for gradient boosted trees
- `HandsML`
    - Boosting
- `ISTL`
    - Chap 8.2.3

### 8. Dimensionality Reduction
8.1 discusses the **curse of dimensionality** and why is it relevent in ML problems. It then explains the **PCA** algorithm and how to interpret the algorithm's results.

8.2 talks about the **Explained Variance Ratio** and how it can be used to tune the PCA algorithm.

8.3 discusses **manifold learning**.
#### 8.1: Principal Component Analysis
- `HandsML`
    - The Curse of Dimensionality
    - PCA - Preserving the Variance
    - Principal Components
    - Projecting Down to d Dimensions, 
    - Using Scikit-Learn
- `ISTL`
    - Chap 6.3.1
    - Chap 10.1, 10.2.1, 10.2.2 
#### 8.2: More on Principal Component Analysis
- `HandsML`
    - Explained Variance Ratio
    - Choosing the Right Number of Dimensions
    - PCA for Compression
    - Incremental PCA
    - Randomized PCA
- `ISTL`
    - Chap 10.2.3
#### 8.3: Manifold Learning
- `HandsML`
    - Manifold Learning

Additional Readings
- `MLIA`
    - 13.1, 13.2
### 9. Clustering Methods
9.1 discusses clustering techniques, and introduces the **k-means clustering** algorithm.

9.2 discusses another algorithm, the **hierarchical clustering** algorithm.
#### 9.1: k-means Clustering
- `ISTL`
    - Chap 10.3.1, 10.3.3
#### 9.2: Hierarchical Clustering
- `ISTL`
    - Chap 10.3.2
- `MLIA`
    - Chap 10.1, 10.2