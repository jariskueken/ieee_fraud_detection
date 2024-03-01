# Info about the models tested for this dataset

## model scores
- all scores where calculated using the average scores of **stratified k-fold-cross-validation with k = 10**
- the following scores where calculated **before** adjusting the CV process to better fit for time series data
    - options:
        - **rolling cv**
        - **blocked cv**
- the **scores** are calculated as **roc_auc scores** where the **predictions** are calculated using **predict_proba**
- this was also run on a simpler dataset neglecting every non numerical column and filling empty values with -999
- **SCORES (SKF)(Base):**
    | Classifier  | Average Score  | Highest Score  | Lowest Score  | Annotation |
    |---|---|---|---| --- |
    | Linear Regression  | 0.0 %  | 0.0 %  | 0.0% | Has no predict_proba |
    | Logistic Regression  | 81.98 %  | 82.94 %  | 80.88 %  | / |
    | Lasso  | 0.0 %  | 0.0 %  | 0.0 %  | Has no predict_proba |
    | Stochastic Gradient Descent | 0.0 % | 0.0 % | 0.0 % | probability estimates not available for hinge loss|
    | K Neighbors | 80.73 % | 81.50 % | 80.20 % | / |
    | SVM (all kernels) | 0.0 % | 0.0 % | 0.0 % | Has no predict_proba |
    | Gaussian Process | 0.0 % | 0.0 % | 0.0 % | Memory Issues (demands 126 GiB for training ) |
    | Decision Tree | 77.05 % | 77.99 % | **75.85 %** | / |
    | Random Forrest | 80.62 % | 81.41 % | 79.42 % | / |
    | AdaBoost | **87.31 %** | **88.07 %** | 86.55 % | / |
    | Gradient Boosting | 0.0 % | 0.0 % | 0.0 % | Runtime |
    | Gaussian Naive Bayes | 68.30 % | 70.09 % | 67.65 % | / |
    | Bernoulli Naive Bayes | 72.07 % | 73.10 % | 70.96 % | / |
    | Linear Discriminant Analysis | 83.88 % | 84.80 % | 82.89 % | / |
    | Quadratic Discriminant Analysis | 81.39 % | 82.69 % | 79.91 % | / |
    | Neural Network | 81.39 % | 82.69 % | 79.91 % | / |

- **SCORES(TSS)(Base):**
    | Classifier  | Average Score  | Highest Score  | Lowest Score  | Annotation |
    |---|---|---|---| --- |
    | Random Forrest(2000estimators, max_depth12) | 85.15  %  |  86.79 %  | 84.15 % | / |
    | AdaBoost  | 85.83 %  | 87.80 %  | 84.48 %  | / |
    | Neural Network  | 80.25 %  | 84.15 %  | 74.36 %  | / |
    | XGBoost  | 92.44 %  | 94.02 % | 89.37 % | / |
    | CatBoost  | 91.81 %  | 93.66 %  | 88.02 %  | / |
    | LGBM  | 91.60 %  | 93.68 %  | 89.21 %  | / |

- **SCORES(Holdout)(Base)**
    | Classifier | Score |
    | --- | --- |
    | VotingEnsemble(XGB, CatBoost, LGBM, RF(md18, ne816, mf3, rs331)) | 95.38 % |
- **SCORES(HV)(Base):**
    | Classifier  | Average Score |
    |---|---|
    | Random Forrest(2000estimators, max_depth12, max_features1, random_state42) | 85.09  %  |
    | Random Forrest(816estimators, max_depth18, max_features3, random_state331) | 88.62  %  |
    | Random Forrest(831estimators, max_depth15, max_features6, random_state1000) | 88.23  %  |
    | Random Forrest(1558estimators, max_depth14, max_features2, random_state47) | 87.51  %  |
    | Random Forrest(507estimators, max_depth12, max_features2, random_state873) | 85.96  %  |

- **Kaggle scores** when training on the entire joined dataset where we **replaced missing values with collumn mean** and **dropped columns with more than 90 % missing values** and **encoding categorical features**:
- **NOTE:** logisitic regression and k-neighbors are not re-evaluated
    | Classifier  | Private Score  | Public Score |
    |---|---|---|
    | Logistic Regression  | 50.29 %  | 50.55 %  |
    | K Neighbors | 50.06 % | 49.93 % |
    | Decision Tree | 55.72 % | 59.83 % |
    | Random Forrest | **62.11 %** | 70.67 % |
    | AdaBoost | **70.89 %** | **74.75 %** |
    | Gaussian Naive Bayes | 63.70 % | 71.84 % |
    | Bernoulli Naive Bayes | 71.85 % | 79.95 % |
    | Linear Discriminant Analysis | 68.59 % | 72.95 % |
    | Quadratic Discriminant Analysis | 50.00 % | 50.00 % |
    | Neural Network | 77.12 % | **81.82 %** |

- moving forward using **Random Forrest, AdaBoost and Neural Network and Bernoulli NB** as they returned the most promissing results

- WRONGLY SCALED DATA!
- patterns in the data aren't necessarily what we want to find but rather finding **anomalies**, CV scores of the three used clfs for **unscaled** data

    | Classifier  | Average Score  | Highest Score  | Lowest Score  | Annotation |
    |---|---|---|---| --- |
    | Random Forrest  | 80.62  %  | 81.41 %  | 79.42 % | **improved average score** |
    | AdaBoost  | 87.31 %  | 88.08 %  | 86.55 %  | **improved average score** |
    | Neural Network  | 49.97 %  | 50.00 %  | 49.77 %  | **big drop in cv score** |
    | Bernoulli NB | 74.38 % | 75.46 % | 73.42 % | improvement in average score

- training and prediciting on unscaled data actually improves the performance scores on kaggle as well

    | Classifier  | Private Score  | Public Score |
    |---|---|---|
    | Decision Tree | 78.55 % | 80.64 % |
    | Random Forrest | **79.48 %** | 85.36 % |
    | AdaBoost | **86.63 %** | **89.04 %** |
    | Bernoulli Naive Bayes | 72.26 % | 80.53 % |
    | Neural Network | 50.00 % | **50.00 %** |

- scores when using **SMOTE** to **resample targets to 50:50** and again using **standard scaled data**

    | Classifier  | Average Score  | Highest Score  | Lowest Score  | Annotation |
    |---|---|---|---| --- |
    | Random Forrest  | 90.60  %  | 90.99 %  | 90.18 % | **improvement** |
    | AdaBoost  | 98.59 %  | 98.64 %  | 98.53 %  | **improvement** |
    | Bernoulli Naive Bayes | 74.53 % | 74.62 % | 74.29 % | **improvement** |
    | Neural Network  | 96.14 %  | 96.34 %  | 95.73 %  | **improvement** |

- **SMOTE** didn't seem to increase (actually decreases) the kaggle score at all so currently this approach is not longer evaluated

- as we seem to heavily overfit **scaling down** the data was one approach
- started with **feature count** scaling for that calculated correlation between features
    - **group features** that have a **correlation coeffitient higher then 0.5 or lower then -0.5**
    - run **pca** on these subset of features (new pca for each of the subsets)
    - take those **principal components out of the first 3** that have a **cumulated explained variance ratio of 90% or above** as new features
    - take the remaining uncorrelated features as normal features as well
- this process resulted in downscaling from **423** features to **77** features

    | Classifier  | Average Score  | Highest Score  | Lowest Score  | Annotation |
    |---|---|---|---| --- |
    | Random Forrest  | 81.67  %  | 83.06 %  | 80.38 % | **similar to base sores** |
    | AdaBoost  | 86.90 %  | 87.56 %  | 86.27 %  | **similar to base scores** |
    | Bernoulli Naive Bayes | 73.55 % | 74.63 % | 72.53 % | **similar to base score** |
    | Neural Network  | 80.08 %  | 81.47 %  | 78.99 %  | **similar to base score** |

- kaggle scores are similar to the base score using this technique