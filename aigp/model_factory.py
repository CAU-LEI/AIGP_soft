# aigp/model_factory.py
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier, ExtraTreesClassifier
import warnings

# 尝试导入 xgboost 模块
try:
    from xgboost import XGBRegressor, XGBClassifier

    xgboost_available = True
except ImportError:
    xgboost_available = False

from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier


def get_model(task_type, model_name, model_params=None, gpu=False, categorical=False):
    """
    根据任务类型和模型名称返回模型实例

    参数：
      task_type: "regression" 或 "classification"
      model_name: 模型名称字符串（例如 "LinearRegression", "LogisticRegression", "CatBoostClassifier" 等）
      model_params: 模型参数字典
      gpu: 是否使用 GPU 训练
      categorical: 是否存在分类变量（若存在，仅允许使用 LightGBM 或 CatBoost）

    返回：
      model: 已初始化的模型实例
    """
    model_params = model_params or {}

    # 如果存在分类变量，则只允许使用 CatBoost 或 LGBM 模型
    if categorical:
        allowed = ["CatBoost", "CatBoostClassifier", "CatBoostRegressor", "LGBM", "LGBMClassifier", "LGBMRegressor"]
        if model_name not in allowed:
            raise ValueError("当存在分类变量时，仅支持 lightgbm 或 catboost 方法！")

    # 根据任务类型构建模型字典
    if task_type == "regression":
        models = {
            "knn": lambda: KNeighborsRegressor(**model_params),
            "svm": lambda: SVR(**model_params),
            "LinearRegression": lambda: LinearRegression(**model_params),
            "RidgeRegression": lambda: Ridge(**model_params),
            "ElasticNet": lambda: ElasticNet(**model_params),
            "RandomForest": lambda: RandomForestRegressor(**model_params),
            "GradientBoosting": lambda: GradientBoostingRegressor(**model_params),
            "AdaBoost": lambda: AdaBoostRegressor(**model_params),
            "CatBoostRegressor": lambda: CatBoostRegressor(**(add_gpu_params(model_params, gpu, model_type="catboost")),
                                                           verbose=0),
            "LGBMRegressor": lambda: LGBMRegressor(**(add_gpu_params(model_params, gpu, model_type="lgbm")))
        }
        if xgboost_available:
            models["xgboost"] = lambda: XGBRegressor(**(add_gpu_params(model_params, gpu, model_type="xgboost")))
        else:
            warnings.warn("xgboost 包未安装，将无法使用 xgboost 模型。")
    elif task_type == "classification":  # 分类任务
        models = {
            "knn": lambda: KNeighborsClassifier(**model_params),
            "svm": lambda: SVC(**model_params),
            "LogisticRegression": lambda: LogisticRegression(**model_params, max_iter=1000),
            "RandomForest": lambda: RandomForestClassifier(**model_params),
            "GradientBoosting": lambda: GradientBoostingClassifier(**model_params),
            "AdaBoost": lambda: AdaBoostClassifier(**model_params),
            "CatBoost": lambda: CatBoostClassifier(**(add_gpu_params(model_params, gpu, model_type="catboost")),
                                                   verbose=0),
            "LGBM": lambda: LGBMClassifier(**(add_gpu_params(model_params, gpu, model_type="lgbm"))),
            "ExtraTrees": lambda: ExtraTreesClassifier(**model_params)
        }
        if xgboost_available:
            models["xgboost"] = lambda: XGBClassifier(**(add_gpu_params(model_params, gpu, model_type="xgboost")))
        else:
            warnings.warn("xgboost 包未安装，将无法使用 xgboost 模型。")
    else:
        raise ValueError("未知任务类型: {}".format(task_type))

    if model_name not in models:
        raise ValueError("不支持的模型名称: {}，可选模型包括: {}".format(model_name, list(models.keys())))

    return models[model_name]()


def add_gpu_params(params, gpu, model_type):
    """
    根据 gpu 参数对模型参数进行更新。

    参数：
      params: 原始参数字典
      gpu: 是否使用 gpu
      model_type: 模型类型，"catboost"、"lgbm" 或 "xgboost"

    返回：
      更新后的参数字典
    """
    params = params.copy()
    if gpu:
        if model_type == "catboost":
            params["task_type"] = "GPU"
        elif model_type == "lgbm":
            params["device"] = "gpu"
        elif model_type == "xgboost":
            params["tree_method"] = "gpu_hist"
    return params
