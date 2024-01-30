from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, \
    LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# list of classifiers to test
CLASSIFIER_NAMES = [
    "Linear Regression",
    "Logistic Regression",
    "Lasso",
    "Stochastic Gradient Descent Classifier",
    "Nearest Neighbors",
    "Linear SVM kernel",
    "RBF SVM",
    "Nu SVM",
    "Linear SVM Model",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Gradient Boosting",
    "Gaussian Naive Bayes",
    "Bernoulli Naive Bayes",
    "Linear Discriminant Analysis",
    "QDA"
]

CLASSIFIER_NAMES_SHORT = [
    "LinReg",
    "LogReg",
    "Lasso",
    "SGD",
    "KNC",
    "LinSVMKern",
    "RBF SVM",
    "NuSVM",
    "LinSVM"
    "GP",
    "DT",
    "RF",
    "NN",
    "AdaBoost",
    "GradBoost",
    "GNB",
    "BNB",
    "LDA",
    "QDA"
]

CLASSIFIERS = [
    LinearRegression(),
    LogisticRegression(),
    Lasso(),
    SGDClassifier(random_state=42),
    KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025, random_state=42),
    # SVC(gamma=2, C=1, random_state=42),
    # NuSVC(random_state=42),
    # LinearSVC(random_state=42),
    # GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(random_state=42),
    # GradientBoostingClassifier(random_state=42),
    GaussianNB(),
    BernoulliNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
]

CLASSIFIERS_DICT_ALL = {
    # "LinReg": LinearRegression(),
    # "LogReg": LogisticRegression(),
    # "Lasso": Lasso(),
    # "SGD": SGDClassifier(random_state=42),
    # "KNC": KNeighborsClassifier(3),
    # "LinSVMKern": SVC(kernel="linear", C=0.025, random_state=42),
    # "RBF_SVM": SVC(gamma=2, C=1, random_state=42),
    # "NuSVM": NuSVC(random_state=42),
    # "LinSVM": LinearSVC(random_state=42),
    # "GP": GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    # "DT": DecisionTreeClassifier(max_depth=5, random_state=42),
    # "RF": RandomForestClassifier(
    #     max_depth=12, n_estimators=2000, max_features=1, random_state=42
    # ),
    # "AdaBoost": AdaBoostClassifier(random_state=42),
    # "GradBoost": GradientBoostingClassifier(random_state=42),
    # "GNB": GaussianNB(),
    # "BNB": BernoulliNB(),
    # "LDA": LinearDiscriminantAnalysis(),
    # "QDA": QuadraticDiscriminantAnalysis(),
    # "NN": MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    # "xgb": XGBClassifier(n_estimators=2000,
    #                      max_depth=12,
    #                      learning_rate=0.02,
    #                      subsample=0.8,
    #                      colsample_bytree=0.4,
    #                      eval_metric='auc'),
    # "cat": CatBoostClassifier(n_estimators=2000,
    #                           max_depth=12,
    #                           learning_rate=0.02,
    #                           subsample=0.8,
    #                           eval_metric='AUC'),
    # "lgbm": LGBMClassifier(n_estimators=2000,
    #                        max_depth=12,
    #                        learning_rate=0.02,
    #                        subsample=0.8,
    #                        colsample_bytree=0.4,
    #                        metric="auc")
}

CLASSIFIERS_DICT = {
    "RF_Base": RandomForestClassifier(
        max_depth=12, n_estimators=2000, max_features=1, random_state=42
    ),
    "RF_1": RandomForestClassifier(
        max_depth=15, n_estimators=831, max_features=6, random_state=1000
    ),
    "RF_2": RandomForestClassifier(
        max_depth=14, n_estimators=1558, max_features=4, random_state=47
    ),
    "RF_3": RandomForestClassifier(
        max_depth=12, n_estimators=507, max_features=2, random_state=873
    ),
    # "AdaBoost": AdaBoostClassifier(random_state=42),
    # "xgb": XGBClassifier(n_estimators=2000,
    #                      max_depth=12,
    #                      learning_rate=0.02,
    #                      subsample=0.8,
    #                      colsample_bytree=0.4,
    #                      eval_metric='auc'),
    # "cat": CatBoostClassifier(n_estimators=2000,
    #                           max_depth=12,
    #                           learning_rate=0.02,
    #                           subsample=0.8,
    #                           eval_metric='AUC'),
    # "lgbm": LGBMClassifier(n_estimators=2000,
    #                        max_depth=12,
    #                        learning_rate=0.02,
    #                        subsample=0.8,
    #                        colsample_bytree=0.4,
    #                        metric="auc")
    "RF": RandomForestClassifier(
        max_depth=18,
        n_estimators=816,
        max_features=3,
        random_state=331
    ),
    "xgb": XGBClassifier(n_estimators=2000,
                         max_depth=12,
                         learning_rate=0.02,
                         subsample=0.8,
                         colsample_bytree=0.4,
                         eval_metric='auc'),
    "cat": CatBoostClassifier(n_estimators=2000,
                              max_depth=12,
                              learning_rate=0.02,
                              subsample=0.8,
                              eval_metric='AUC',
                              verbose=False),
    "lgbm": LGBMClassifier(n_estimators=2000,
                           max_depth=12,
                           learning_rate=0.02,
                           subsample=0.8,
                           colsample_bytree=0.4,
                           metric="auc")
}

CLFS_SHORT_DICT = {
    "SGD": SGDClassifier(random_state=42),
    "NN": MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    "LogReg": LogisticRegression(),
    "NuSVM": NuSVC(random_state=42),
    "RBF_SVM": SVC(gamma=2, C=1, random_state=42)
}
