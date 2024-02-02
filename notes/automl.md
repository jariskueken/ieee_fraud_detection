# AutoGluon

- running merged training dataset with already some FE done beforehand eg:
    - fill missing vals with 999
    - drop columns with more than 90% empty
    - merge datasets
    - encode categorical features

- also already specified eval metric beforehand as roc_auc

- running on different timelimits:
    - 120 -> kaggle score:
    - 1200 -> kaggle score:
    - 7200 -> kaggle score:

- running presets: best_quality -> requires more than 120 seconds trying on 240 first
    - 240 fails as it decides on no best model to predict on, trying on 1200 next
    - runs on 1200

- adding a validation set, running a train/validation split on the training set with 70/30
    - need to set user_bag_holdout=True in order to work

- further improvements to test following the docs: https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-kaggle.html
    - use hyperparameter presets: 'best_quality'
    - as we have time dependent samples use a validation set