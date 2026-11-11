from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

from ..core.adapters import SklearnClusterer
from ..core.registry import CLUSTERERS


@CLUSTERERS.register("kmeans")
class KMeansModule(SklearnClusterer):
    estimator_cls = KMeans
    default_params = {"n_clusters": 3, "n_init": 10, "random_state": 42}


@CLUSTERERS.register("dbscan")
class DBSCANModule(SklearnClusterer):
    estimator_cls = DBSCAN
    default_params = {"eps": 0.5, "min_samples": 5}


@CLUSTERERS.register("hierarchical")
class HierarchicalModule(SklearnClusterer):
    estimator_cls = AgglomerativeClustering
    default_params = {"n_clusters": 3, "linkage": "ward"}


@CLUSTERERS.register("gmm")
class GMMModule(SklearnClusterer):
    estimator_cls = GaussianMixture
    default_params = {"n_components": 3, "random_state": 42}

    def fit_predict(self, X):
        return self.model.fit(X).predict(X)
