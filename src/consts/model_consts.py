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

CLASSIFIERS_DICT = {
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
    "RF": RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    # "GradBoost": GradientBoostingClassifier(random_state=42),
    # "GNB": GaussianNB(),
    # "BNB": BernoulliNB(),
    # "LDA": LinearDiscriminantAnalysis(),
    # "QDA": QuadraticDiscriminantAnalysis(),
    "NN": MLPClassifier(alpha=1, max_iter=1000, random_state=42),
}

CLFS_SHORT_DICT = {
    "SGD": SGDClassifier(random_state=42),
    "NN": MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    "LogReg": LogisticRegression(),
    "NuSVM": NuSVC(random_state=42),
    "RBF_SVM": SVC(gamma=2, C=1, random_state=42)
}
