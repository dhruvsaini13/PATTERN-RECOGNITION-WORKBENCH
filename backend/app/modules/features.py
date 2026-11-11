from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..core.adapters import SklearnExtractor
from ..core.registry import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register("standard_scaler")
class StandardScalerModule(SklearnExtractor):
    estimator_cls = StandardScaler
    default_params = {}


@FEATURE_EXTRACTORS.register("minmax_scaler")
class MinMaxScalerModule(SklearnExtractor):
    estimator_cls = MinMaxScaler
    default_params = {}


@FEATURE_EXTRACTORS.register("pca")
class PCAModule(SklearnExtractor):
    estimator_cls = PCA
    default_params = {"n_components": 2}


@FEATURE_EXTRACTORS.register("lda")
class LDAModule(SklearnExtractor):
    estimator_cls = LinearDiscriminantAnalysis
    default_params = {"n_components": 2}


@FEATURE_EXTRACTORS.register("select_k_best")
class SelectKBestModule(SklearnExtractor):
    estimator_cls = SelectKBest
    default_params = {"score_func": f_classif, "k": 5}
