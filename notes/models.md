# Info about the models tested for this dataset

## model scores
- all scores where calculated using the average scores of **stratified k-fold-cross-validation with k = 10**
- the following scores where calculated **before** adjusting the CV process to better fit for time series data
    - options:
        - **rolling cv**
        - **blocked cv**
- the **scores** are calculated as **roc_auc scores** where the **predictions** are calculated using **predict_proba**
- this was also run on a simpler dataset neglecting every non numerical column and filling empty values with the mean of the corresponding collumn
- **SCORES:**
    | Classifier  | Average Score  | Highest Score  | Lowest Score  | Annotation |
    |---|---|---|---| --- |
    | Linear Regression  | 0.0 %  | 0.0 %  | 0.0% | Has no predict_proba |
    | Logistic Regression  | 85.57 %  | 86.22 %  | 84.65 %  | / |
    | Lasso  | 0.0 %  | 0.0 %  | 0.0 %  | Has no predict_proba |
    | Stochastic Gradient Descent | 0.0 % | 0.0 % | 0.0 % | probability estimates not available for hinge loss|
    | K Neighbors | 83.86 % | 84.97 % | 83.13 % | / |
    | SVM (all kernels) | 0.0 % | 0.0 % | 0.0 % | Has no predict_proba |
    | Gaussian Process | 0.0 % | 0.0 % | 0.0 % | Memory Issues (demands 126 GiB for training ) |
    | Decision Tree | 76.84 % | 77.81 % | **74.26 %** | / |
    | Random Forrest | 79.97 % | 81.14 % | 78.85 % | / |
    | AdaBoost | **87.56 %** | **88.42 %** | 87.14 % | / |
    | Gradient Boosting | 0.0 % | 0.0 % | 0.0 % | Runtime |
    | Gaussian Naive Bayes | 74.92 % | 75.70 % | 74.05 % | / |
    | Bernoulli Naive Bayes | 74.69 % | 75.73 % | 73.76 % | / |
    | Linear Discriminant Analysis | 84.19 % | 84.89 % | 83.64 % | / |
    | Quadratic Discriminant Analysis | 80.70 % | 82.22 % | 79.66 % | / |
    | Neural Network | 85.31 % | 86.52 % | 84.51 % | / |

- most models **overfit**, should be result of the imbalanced dataset we have

- **Kaggle scores** when training on the entire joined dataset where we **replaced missing values with collumn mean** and **dropped columns with more than 90 % missing values** and **encoding categorical features**:
    | Classifier  | Private Score  | Public Score |
    |---|---|---|
    | Logistic Regression  | 50.29 %  | 50.55 %  |
    | K Neighbors | 50.06 % | 49.93 % |
    | Decision Tree | 50.23 % | 49.88 % |
    | Random Forrest | **52.28 %** | 49.90 % |
    | AdaBoost | **50.51 %** | **50.36 %** |
    | Gaussian Naive Bayes | 50.85 % | 50.19 % |
    | Bernoulli Naive Bayes | 50.12 % | 50.20 % |
    | Linear Discriminant Analysis | 50.64 % | 50.23 % |
    | Quadratic Discriminant Analysis | 80.70 % | 82.22 % | 79.66 % | / |
    | Neural Network | 85.31 % | 86.52 % | 84.51 % | / |