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
    | Logistic Regression  | 100.0 %  | 100.0 %  | 100.0 %  | / |
    | Lasso  | 0.0 %  | 0.0 %  | 0.0 %  | Has no predict_proba |
    | Stochastic Gradient Descent | 0.0 % | 0.0 % | 0.0 % | probability estimates not available for hinge loss|
    | K Neighbors | 97.449 % | 98.064 % | 97.111 % | / |
    | SVM (all kernels) | 0.0 % | 0.0 % | 0.0 % | Has no predict_proba |
    | Gaussian Process | 0.0 % | 0.0 % | 0.0 % | Memory Issues (demands 126 GiB for training ) |
    | Decision Tree | 100.0 % | 100.0 % | 100.0 % | / |
    | Random Forrest | 84.006 % | 85.118 % | 81.904 % | / |
    | AdaBoost | 100.0 % | 100.0 % | 100.0 % | / |
    | Gradient Boosting | 100.0 % | 100.0 % | 100.0 % | Runtime 923 seconds |
    | Gaussian Naive Bayes | 100.0 % | 100.0 % | 100.0 % | / |
    | Bernoulli Naive Bayes | 88.684 % | 89.353 % | 87.314 % | / |
    | Linear Discriminant Analysis | 100.0 % | 100.0 % | 100.0 % | / |
    | Quadratic Discriminant Analysis | 99.982 % | 100.0 % | 99.956 % | / |
    | Neural Network | 100.0 % | 99.999.. % | 99.999.. % | / |

- most models **overfit**, should be result of the imbalanced dataset we have