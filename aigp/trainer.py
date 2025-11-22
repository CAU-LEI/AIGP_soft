# aigp/trainer.py
import time
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
from .utils import Timer
from .model_factory import get_model
import os
def save_feature_importance(model, X, output_prefix="feature_importance"):
    """
    输出模型训练后的特征重要性：
      - 如果模型具有 feature_importances_ 或 coef_ 属性，则提取特征重要性；
      - 将结果保存为 CSV 文件，并生成条形图保存为 PNG 文件。
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    feature_names = X.columns if hasattr(X, "columns") else [f"Feature_{i}" for i in range(X.shape[1])]
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = model.coef_
        if importance.ndim > 1:
            importance = abs(importance).mean(axis=0)
        else:
            importance = abs(importance)
    else:
        print("当前模型不支持内置特征重要性输出。")
        return
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importance})
    csv_file = output_prefix + ".csv"
    df_imp.to_csv(csv_file, index=False)
    print("特征重要性已保存到", csv_file)
    plt.figure(figsize=(10, 6))
    df_imp.sort_values(by="importance", ascending=False, inplace=True)
    plt.barh(df_imp["feature"], df_imp["importance"])
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    png_file = output_prefix + ".png"
    plt.savefig(png_file)
    print("特征重要性图已保存到", png_file)
    plt.close()







def pearson_corr(y_true, y_pred):
    """计算皮尔逊相关系数"""
    if len(y_true) < 2:
        return 0
    corr, _ = pearsonr(y_true, y_pred)
    return corr


def get_regression_scorer():
    """返回皮尔逊相关系数 scorer（回归任务）"""
    return make_scorer(pearson_corr, greater_is_better=True)


def run_cross_validation(model, X, y, cv, task_type, n_jobs=1, random_state=42, output_file=None):
    """使用交叉验证评估模型性能"""
    print("Running {}-fold cross validation...".format(cv))
    start = time.time()
    scorer = get_regression_scorer() if task_type == "regression" else "accuracy"
    
    # 创建KFold对象确保每折结果一致
    from sklearn.model_selection import KFold, StratifiedKFold
    if task_type == "classification":
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    scores = cross_val_score(model, X, y, cv=kfold, scoring=scorer, n_jobs=n_jobs)
    elapsed = time.time() - start
    
    # 准备输出内容
    output_lines = []
    output_lines.append("Cross validation time: {:.2f} sec".format(elapsed))
    output_lines.append("")
    output_lines.append("=== {}-Fold Cross Validation Results ===".format(cv))
    
    # 输出每折的结果
    for i, score in enumerate(scores, 1):
        line = f"Fold {i:2d}: {score:.4f}"
        print(line)
        output_lines.append(line)
    
    # 计算统计量
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    output_lines.append("-" * 40)
    output_lines.append(f"Mean:     {mean_score:.4f}")
    output_lines.append(f"Std Dev:  {std_score:.4f}")
    output_lines.append(f"Min:      {np.min(scores):.4f}")
    output_lines.append(f"Max:      {np.max(scores):.4f}")
    output_lines.append("=" * 40)
    
    # 打印到控制台
    print("Cross validation time: {:.2f} sec".format(elapsed))
    print("\n=== {}-Fold Cross Validation Results ===".format(cv))
    print("-" * 40)
    print(f"Mean:     {mean_score:.4f}")
    print(f"Std Dev:  {std_score:.4f}")
    print(f"Min:      {np.min(scores):.4f}")
    print(f"Max:      {np.max(scores):.4f}")
    print("=" * 40)
    
    # 写入文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"结果已保存到文件: {output_file}")
    
    return scores, mean_score


# def run_train_test(model, X, y, train_size=None, ntest=None, task_type="regression"):
#     """对数据进行训练/测试划分，并训练模型与评估"""
#     if train_size is None and ntest is None:
#         raise ValueError("Must specify train_size or ntest!")
#
#     if train_size is not None:
#         X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
#     else:
#         X_train = X.iloc[:ntest, :]
#         y_train = y.iloc[:ntest]
#         X_test = X.iloc[ntest:, :]
#         y_test = y.iloc[ntest:]
#
#     print("Training samples:", len(X_train))
#     print("Testing samples:", len(X_test))
#
#     with Timer("Model training"):
#         model.fit(X_train, y_train)
#     with Timer("Model prediction"):
#         y_pred = model.predict(X_test)
#
#     if task_type == "regression":
#         score = pearson_corr(y_test, y_pred)
#         print("Pearson correlation:", score)
#     else:
#         score = accuracy_score(y_test, y_pred)
#         print("Accuracy:", score)
#     return model, score

import pandas as pd
import random

def run_train_test(model, X, y, train_size=None, ntest=None, task_type="regression", random_state=42):
    """对数据进行训练/测试划分，并训练模型与评估"""
    if train_size is None and ntest is None:
        raise ValueError("Must specify train_size or ntest!")

    if train_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)
    else:
        X_train = X.iloc[:ntest, :]
        y_train = y.iloc[:ntest]
        X_test = X.iloc[ntest:, :]
        y_test = y.iloc[ntest:]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    with Timer("Model training"):
        model.fit(X_train, y_train)
    with Timer("Model prediction"):
        y_pred = model.predict(X_test)

    # ✅ 计算分数
    if task_type == "regression":
        score = pearson_corr(y_test, y_pred)
        print("Pearson correlation:", score)
    else:
        score = accuracy_score(y_test, y_pred)
        print("Accuracy:", score)

    # ✅ 保存预测结果
    df_pred = pd.DataFrame({
        "sample_id": X_test.index,
        "true_value": y_test.values,
        "pred_value": y_pred
    })
    df_pred.to_csv("test_predictions.csv", index=False)
    print("测试集预测结果已保存到 test_predictions.csv")

    return model, score



def run_grid_search(model, X, y, grid_params, cv, task_type, n_jobs=1, random_state=42):
    """使用网格搜索调参"""
    print("Starting grid search...")
    scorer = get_regression_scorer() if task_type == "regression" else "accuracy"
    
    # 创建KFold对象确保每折结果一致
    from sklearn.model_selection import KFold, StratifiedKFold
    if task_type == "classification":
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    gs = GridSearchCV(estimator=model, param_grid=grid_params, cv=kfold, scoring=scorer, n_jobs=n_jobs)
    with Timer("Grid search"):
        gs.fit(X, y)
    print("Best parameters:", gs.best_params_)
    print("Best score:", gs.best_score_)
    return gs.best_estimator_, gs.best_score_, gs.best_params_


def run_ssa_search(model, X, y, ssa_params, cv, task_type, n_jobs=1):
    """
    使用 SSA (麻雀搜索) 进行超参数调参：
    仅支持 LightGBM 和 CatBoost 模型（回归和分类任务均支持）。
    如果 ssa_params 中设置 "use_custom_ssa": true，则调用定制 SSA 版本。
    """
    supported_models = ["LGBMRegressor", "LGBMClassifier", "CatBoostRegressor", "CatBoostClassifier"]
    if ssa_params.get("use_custom_ssa", False) and model.__class__.__name__ in supported_models:
        if model.__class__.__name__ in ["LGBMRegressor", "LGBMClassifier"]:
            best_params, best_metric = run_ssa_search_lgbm(X, y, ssa_params, n_jobs, task_type, cv)
        elif model.__class__.__name__ in ["CatBoostRegressor", "CatBoostClassifier"]:
            best_params, best_metric = run_ssa_search_catboost(X, y, ssa_params, n_jobs, task_type)
        
        # 确保整数参数被正确转换
        if model.__class__.__name__ in ["LGBMRegressor", "LGBMClassifier"]:
            best_params['num_leaves'] = int(best_params['num_leaves'])
            if 'max_depth' in best_params:
                best_params['max_depth'] = int(best_params['max_depth'])
        elif model.__class__.__name__ in ["CatBoostRegressor", "CatBoostClassifier"]:
            if 'depth' in best_params:
                best_params['depth'] = int(best_params['depth'])
        
        best_model = get_model(task_type, model.__class__.__name__, model_params=best_params, gpu=False,
                               categorical=False)
        with Timer("Training best model (SSA)"):
            best_model.fit(X, y)
        
        # 使用交叉验证评估最终模型
        if cv:
            import numpy as np
            print(f"\n=== 最终模型 {cv}折交叉验证结果 ===")
            scores, avg_score = run_cross_validation(best_model, X, y, cv, task_type, n_jobs=n_jobs, random_state=42)
            extra_info["cv_scores"] = scores
            extra_info["cv_avg_score"] = avg_score
            print(f"CV平均得分: {avg_score:.6f}")
            print(f"CV得分标准差: {np.std(scores):.6f}")
        
        print("SSA best parameters:", best_params)
        print("SSA best metric:", best_metric)
        return best_model, best_metric, best_params
    else:
        # 使用通用 SSA 实现（伪实现）
        import random
        iterations = ssa_params.get("iterations", 10)
        param_grid = ssa_params.get("param_grid", {})
        candidate_params = []
        for _ in range(iterations):
            candidate = {}
            for key, values in param_grid.items():
                candidate[key] = random.choice(values)
            candidate_params.append(candidate)
        from joblib import Parallel, delayed
        def evaluate_candidate(params):
            m = get_model(task_type, model.__class__.__name__, model_params=params, gpu=False, categorical=False)
            scores, avg_score = run_cross_validation(m, X, y, cv=cv or 3, task_type=task_type, n_jobs=n_jobs)
            return params, avg_score

        results = Parallel(n_jobs=n_jobs)(delayed(evaluate_candidate)(params) for params in candidate_params)
        if task_type == "regression":
            best_score = float('inf')
            best_params = None
            for params, score in results:
                if score < best_score:
                    best_score = score
                    best_params = params
        else:
            best_score = -1e9
            best_params = None
            for params, score in results:
                if score > best_score:
                    best_score = score
                    best_params = params
        best_model = get_model(task_type, model.__class__.__name__, model_params=best_params, gpu=False,
                               categorical=False)
        with Timer("Training best model (Generic SSA)"):
            best_model.fit(X, y)
        print("Generic SSA best parameters:", best_params)
        print("Generic SSA best score:", best_score)
        return best_model, best_score, best_params


def run_ssa_search_lgbm(X, y, ssa_params, n_jobs, task_type, cv=3):
    """
    针对 LightGBM 的改进SSA搜索实现，支持回归和分类任务。
    回归任务使用 RMSE 作为目标；分类任务使用负准确率作为目标。
    """
    from sklearn.model_selection import train_test_split, cross_val_score
    import lightgbm as lgb
    import numpy as np
    import random
    
    # 使用交叉验证作为目标函数
    def objective_function(params):
        try:
            if task_type == "regression":
                model = lgb.LGBMRegressor(
                    learning_rate=params['learning_rate'],
                    num_leaves=int(params['num_leaves']),
                    max_depth=int(params['max_depth']),
                    n_estimators=100,
                    random_state=42,
                    verbose=-1
                )
                scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=n_jobs)
                return -np.mean(scores)  # 返回MSE
            else:
                model = lgb.LGBMClassifier(
                    learning_rate=params['learning_rate'],
                    num_leaves=int(params['num_leaves']),
                    max_depth=int(params['max_depth']),
                    n_estimators=100,
                    random_state=42,
                    verbose=-1
                )
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=n_jobs)
                return -np.mean(scores)  # 返回负准确率
        except:
            return float('inf')

    param_bounds = ssa_params.get("param_bounds", {
        'learning_rate': (0.01, 0.3),
        'num_leaves': (10, 100),
        'max_depth': (3, 15)
    })
    pop_size = ssa_params.get("pop_size", 20)
    max_iter = ssa_params.get("max_iter", 30)
    discoverer_ratio = ssa_params.get("discoverer_ratio", 0.2)
    scouter_ratio = ssa_params.get("scouter_ratio", 0.1)

    class ImprovedSparrowSearch:
        def __init__(self, obj_func, param_bounds, pop_size=20, max_iter=30, 
                     discoverer_ratio=0.2, scouter_ratio=0.1):
            self.obj_func = obj_func
            self.param_bounds = param_bounds
            self.pop_size = pop_size
            self.max_iter = max_iter
            self.discoverer_ratio = discoverer_ratio
            self.scouter_ratio = scouter_ratio
            
            self.param_names = list(param_bounds.keys())
            self.dim = len(self.param_names)
            self.lb = np.array([param_bounds[name][0] for name in self.param_names])
            self.ub = np.array([param_bounds[name][1] for name in self.param_names])
            
            self.discoverer_num = int(pop_size * discoverer_ratio)
            self.scouter_num = int(pop_size * scouter_ratio)
            self.follower_num = pop_size - self.discoverer_num - self.scouter_num
            
            self.best_solution = None
            self.best_fitness = float('inf')
            self.fitness_history = []

        def initialize_population(self):
            pop = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):
                for j in range(self.dim):
                    pop[i, j] = self.lb[j] + (self.ub[j] - self.lb[j]) * random.random()
            return pop

        def evaluate_fitness(self, pop):
            fitness = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                params = dict(zip(self.param_names, pop[i]))
                try:
                    fitness[i] = self.obj_func(params)
                except:
                    fitness[i] = float('inf')
            return fitness

        def update_discoverers(self, pop, fitness, iter_num):
            sorted_indices = np.argsort(fitness)
            discoverer_indices = sorted_indices[:self.discoverer_num]
            
            for i in discoverer_indices:
                if random.random() < 0.8:  # 安全状态
                    for j in range(self.dim):
                        noise = np.random.normal(0, 0.1) * (self.ub[j] - self.lb[j])
                        pop[i, j] += noise
                else:  # 危险状态
                    for j in range(self.dim):
                        step = np.random.normal(0, 0.3) * (self.ub[j] - self.lb[j])
                        pop[i, j] += step
                
                pop[i] = np.clip(pop[i], self.lb, self.ub)
            return pop

        def update_followers(self, pop, fitness, iter_num):
            sorted_indices = np.argsort(fitness)
            follower_indices = sorted_indices[self.discoverer_num:self.discoverer_num + self.follower_num]
            best_discoverer = sorted_indices[0]
            
            for i in follower_indices:
                if i > self.pop_size // 2:
                    for j in range(self.dim):
                        step = np.random.normal(0, 0.1) * (self.ub[j] - self.lb[j])
                        pop[i, j] += step
                else:
                    for j in range(self.dim):
                        step = (pop[best_discoverer, j] - pop[i, j]) * random.random()
                        pop[i, j] += step
                
                pop[i] = np.clip(pop[i], self.lb, self.ub)
            return pop

        def update_scouters(self, pop, fitness, iter_num):
            sorted_indices = np.argsort(fitness)
            scouter_indices = sorted_indices[-self.scouter_num:]
            best_individual = sorted_indices[0]
            
            for i in scouter_indices:
                if fitness[i] > np.mean(fitness):
                    for j in range(self.dim):
                        step = (pop[best_individual, j] - pop[i, j]) * random.random()
                        pop[i, j] += step
                else:
                    for j in range(self.dim):
                        step = np.random.normal(0, 0.05) * (self.ub[j] - self.lb[j])
                        pop[i, j] += step
                
                pop[i] = np.clip(pop[i], self.lb, self.ub)
            return pop

        def optimize(self):
            pop = self.initialize_population()
            fitness = self.evaluate_fitness(pop)
            
            best_idx = np.argmin(fitness)
            self.best_solution = pop[best_idx].copy()
            self.best_fitness = fitness[best_idx]
            self.fitness_history.append(self.best_fitness)
            
            for iter_num in range(self.max_iter):
                pop = self.update_discoverers(pop, fitness, iter_num)
                pop = self.update_followers(pop, fitness, iter_num)
                pop = self.update_scouters(pop, fitness, iter_num)
                
                fitness = self.evaluate_fitness(pop)
                
                current_best_idx = np.argmin(fitness)
                if fitness[current_best_idx] < self.best_fitness:
                    self.best_solution = pop[current_best_idx].copy()
                    self.best_fitness = fitness[current_best_idx]
                
                self.fitness_history.append(self.best_fitness)
                
                if (iter_num + 1) % 10 == 0:
                    print(f"SSA迭代 {iter_num + 1}/{self.max_iter}, 当前最优: {self.best_fitness:.6f}")
            
            best_params = dict(zip(self.param_names, self.best_solution))
            return best_params, self.best_fitness

    ssa = ImprovedSparrowSearch(objective_function, param_bounds, pop_size, max_iter, 
                               discoverer_ratio, scouter_ratio)
    best_params, best_metric = ssa.optimize()
    return best_params, best_metric


def run_ssa_search_catboost(X, y, ssa_params, n_jobs, task_type):
    """
    针对 CatBoost 的 SSA 搜索实现，支持回归和分类任务。
    回归任务使用 RMSE 作为目标；分类任务使用负准确率作为目标。
    """
    from sklearn.model_selection import train_test_split
    import numpy as np
    if task_type == "regression":
        from sklearn.metrics import mean_squared_error
        def objective_function(params):
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(
                learning_rate=params['learning_rate'],
                depth=int(params['depth']),
                iterations=int(params['iterations']),
                verbose=0
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            return rmse
    else:
        from sklearn.metrics import accuracy_score
        def objective_function(params):
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(
                learning_rate=params['learning_rate'],
                depth=int(params['depth']),
                iterations=int(params['iterations']),
                verbose=0
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            return -acc

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    param_bounds = ssa_params.get("param_bounds", {
        'learning_rate': (0.01, 0.3),
        'depth': (3, 10),
        'iterations': (100, 500)
    })
    pop_size = ssa_params.get("pop_size", 20)
    max_iter = ssa_params.get("max_iter", 30)

    class SparrowSearch:
        def __init__(self, obj_func, param_bounds, pop_size=20, max_iter=30):
            self.obj_func = obj_func
            self.param_bounds = param_bounds
            self.pop_size = pop_size
            self.max_iter = max_iter
            self.dim = len(param_bounds)
            self.lb = np.array([param_bounds[k][0] for k in param_bounds])
            self.ub = np.array([param_bounds[k][1] for k in param_bounds])

        def optimize(self):
            pop = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
            fitness = np.array([self.obj_func(dict(zip(self.param_bounds.keys(), p))) for p in pop])
            for t in range(self.max_iter):
                best_idx = np.argmin(fitness)
                worst_idx = np.argmax(fitness)
                for i in range(self.pop_size):
                    r1 = np.random.rand()
                    if r1 < 0.8:
                        pop[i] += np.random.randn(self.dim) * (pop[best_idx] - pop[i])
                    else:
                        pop[i] += np.random.randn(self.dim)
                for i in range(self.pop_size):
                    if i == worst_idx:
                        pop[i] += np.random.randn(self.dim) * (pop[best_idx] - pop[i])
                pop = np.clip(pop, self.lb, self.ub)
                fitness = np.array([self.obj_func(dict(zip(self.param_bounds.keys(), p))) for p in pop])
            best_idx = np.argmin(fitness)
            best_params = dict(zip(self.param_bounds.keys(), pop[best_idx]))
            return best_params, np.min(fitness)

    ssa = SparrowSearch(objective_function, param_bounds, pop_size=pop_size, max_iter=max_iter)
    best_params, best_metric = ssa.optimize()
    return best_params, best_metric


from sklearn.model_selection import ParameterGrid

def train_model(model, X, y, task_type, cv=None, train_size=None, ntest=None,
                grid=False, grid_params=None, ssa=False, ssa_params=None, n_jobs=1, save_checkpoint="", random_state=42, result_file=None):
    """
    综合训练函数，支持交叉验证、网格搜索、SSA 搜索，
    并根据传入的参数进行训练集划分（通过 train_size 或 ntest）。
    如果 save_checkpoint 非空，则在训练结束后保存模型到指定路径。
    """
    extra_info = {}
    score = None  # 初始化输出值

    # --------------------------
    # 🧠 网格搜索逻辑
    # --------------------------
    if grid and grid_params is not None:
        if cv:
            # 使用交叉验证的网格搜索
            model, best_score, best_params = run_grid_search(
                model, X, y, grid_params, cv=cv, task_type=task_type, n_jobs=n_jobs, random_state=random_state
            )
            extra_info["best_params"] = best_params
            extra_info["best_score"] = best_score
        else:
            # 没有CV，必须有train_size或ntest
            if train_size is None and ntest is None:
                raise ValueError("Grid search without CV requires train_size or ntest.")

            print("[INFO] Grid search without cross-validation, using fixed train/test split.")
            best_score = -float("inf")
            best_params = None
            best_model = None

            for i, params in enumerate(ParameterGrid(grid_params)):
                print(f"[GRID] Trying params {i + 1}/{len(grid_params)}: {params}")
                candidate_model = get_model(task_type, model.__class__.__name__, model_params=params)
                candidate_model, candidate_score = run_train_test(
                    candidate_model, X, y, train_size, ntest, task_type, random_state
                )
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_params = params
                    best_model = candidate_model

            model = best_model
            score = best_score
            extra_info["best_params"] = best_params
            extra_info["best_score"] = best_score

    # --------------------------
    # 🧠 SSA搜索逻辑
    # --------------------------
    elif ssa and ssa_params is not None:
        model, best_score, best_params = run_ssa_search(
            model, X, y, ssa_params, cv=cv or 3, task_type=task_type, n_jobs=n_jobs
        )
        extra_info["best_params"] = best_params
        extra_info["best_score"] = best_score

    # --------------------------
    # 🧠 模型训练逻辑（无grid/ssa时）
    # --------------------------
    if not grid and not ssa:
        if cv:
            scores, avg_score = run_cross_validation(model, X, y, cv, task_type, n_jobs=n_jobs, random_state=random_state, output_file=result_file)
            extra_info["cv_scores"] = scores
            extra_info["cv_avg_score"] = avg_score
            with Timer("Training on full data"):
                model.fit(X, y)
            score = avg_score
        else:
            model, score = run_train_test(model, X, y, train_size, ntest, task_type, random_state)

    # --------------------------
    # ✅ 模型保存
    # --------------------------
    if save_checkpoint:
        print("准备保存模型检查点到:", save_checkpoint)
        dir_name = os.path.dirname(save_checkpoint)
        if dir_name and not os.path.exists(dir_name):
            print("目录不存在，创建目录:", dir_name)
            os.makedirs(dir_name)
        try:
            import joblib
            joblib.dump(model, save_checkpoint)
            print("Model checkpoint saved to:", save_checkpoint)
        except Exception as e:
            print("保存模型时出错:", e)

    return model, score, extra_info

