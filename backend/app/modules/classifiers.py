from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ..core.adapters import SklearnClassifier
from ..core.registry import CLASSIFIERS


@CLASSIFIERS.register("knn")
class KNN(SklearnClassifier):
    estimator_cls = KNeighborsClassifier
    default_params = {"n_neighbors": 5}


@CLASSIFIERS.register("naive_bayes")
class NaiveBayes(SklearnClassifier):
    estimator_cls = GaussianNB
    default_params = {}


@CLASSIFIERS.register("svm")
class SVM(SklearnClassifier):
    estimator_cls = SVC
    default_params = {"C": 1.0, "kernel": "rbf", "probability": True}


@CLASSIFIERS.register("decision_tree")
class DecisionTree(SklearnClassifier):
    estimator_cls = DecisionTreeClassifier
    default_params = {"max_depth": None}


@CLASSIFIERS.register("random_forest")
class RandomForest(SklearnClassifier):
    estimator_cls = RandomForestClassifier
    default_params = {"n_estimators": 100, "random_state": 42}


@CLASSIFIERS.register("logistic_regression")
class LogReg(SklearnClassifier):
    estimator_cls = LogisticRegression
    default_params = {"max_iter": 1000}
