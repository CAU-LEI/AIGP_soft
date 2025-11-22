# aigp/dim_reduction.py
from sklearn.decomposition import PCA
import phate


def reduce_dimensions(X, method, n_components):
    """
    对数据 X 进行降维

    参数：
      X: 特征数据，DataFrame 或 ndarray
      method: 降维方法，"pca" 或 "phate"
      n_components: 降维后的维度数

    返回：
      X_reduced: 降维后的数据
    """
    if method == "pca":
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
    elif method == "phate":
        reducer = phate.PHATE(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
    else:
        raise ValueError("不支持的降维方法: {}".format(method))
    return X_reduced
