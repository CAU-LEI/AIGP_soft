# aigp/auto_predictor.py
"""
自动化基因组预测模块
提供一键式模型选择、优化和比较功能
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from .model_factory import get_model
from .trainer import run_ssa_search, run_grid_search
from .dim_reduction import reduce_dimensions


class AutoGenomicPredictor:
    """自动化基因组预测器"""
    
    def __init__(self, task_type="regression", cv=5, n_jobs=1, random_state=42):
        self.task_type = task_type
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.results = {}
        
    def get_candidate_models(self):
        """获取候选模型列表"""
        if self.task_type == "regression":
            return {
                "LightGBM": {"class": "LGBMRegressor", "priority": 1},
                "CatBoost": {"class": "CatBoostRegressor", "priority": 1},
                "XGBoost": {"class": "xgboost", "priority": 2},
                "RandomForest": {"class": "RandomForest", "priority": 2},
                "GradientBoosting": {"class": "GradientBoosting", "priority": 3},
                "SVM": {"class": "svm", "priority": 3},
                "KNN": {"class": "knn", "priority": 4},
                "Ridge": {"class": "RidgeRegression", "priority": 4},
                "LinearRegression": {"class": "LinearRegression", "priority": 4},
                "ElasticNet": {"class": "ElasticNet", "priority": 4},
                "AdaBoost": {"class": "AdaBoost", "priority": 4}
            }
        else:  # classification
            return {
                "LightGBM": {"class": "LGBM", "priority": 1},
                "CatBoost": {"class": "CatBoost", "priority": 1},
                "XGBoost": {"class": "xgboost", "priority": 2},
                "RandomForest": {"class": "RandomForest", "priority": 2},
                "GradientBoosting": {"class": "GradientBoosting", "priority": 3},
                "SVM": {"class": "svm", "priority": 3},
                "KNN": {"class": "knn", "priority": 4},
                "LogisticRegression": {"class": "LogisticRegression", "priority": 4},
                "AdaBoost": {"class": "AdaBoost", "priority": 4},
                "ExtraTrees": {"class": "ExtraTrees", "priority": 4}
            }
    
    def get_preprocessing_options(self, n_features):
        """获取数据预处理选项"""
        options = {"none": None}
        
        # 根据特征数选择合适的降维方法
        if n_features > 1000:
            options["pca_100"] = {"method": "pca", "n_components": 100}
            options["pca_200"] = {"method": "pca", "n_components": 200}
        if n_features > 500:
            options["pca_50"] = {"method": "pca", "n_components": 50}
        if n_features > 100:
            options["phate_50"] = {"method": "phate", "n_components": 50}
            
        return options
    
    def evaluate_model(self, model, X, y, model_name):
        """评估单个模型"""
        try:
            # 交叉验证评估
            if self.task_type == "regression":
                # 使用皮尔逊相关系数
                from sklearn.metrics import make_scorer
                from scipy.stats import pearsonr
                
                def pearson_corr(y_true, y_pred):
                    if len(y_true) < 2:
                        return 0
                    corr, _ = pearsonr(y_true, y_pred)
                    return corr
                
                scorer = make_scorer(pearson_corr, greater_is_better=True)
            else:
                scorer = 'accuracy'
                
            scores = cross_val_score(
                model, X, y, cv=self.cv, scoring=scorer, 
                n_jobs=self.n_jobs
            )
            
            if self.task_type == "regression":
                print(f"    ✅ {model_name} 评估成功: 皮尔逊相关系数 {scores.mean():.6f} ± {scores.std():.6f}")
            else:
                print(f"    ✅ {model_name} 评估成功: 准确率 {scores.mean():.6f} ± {scores.std():.6f}")
            return {
                'model_name': model_name,
                'cv_scores': scores,
                'cv_mean': scores.mean(),
                'cv_std': scores.std(),
                'model': model,
                'status': 'success'
            }
        except Exception as e:
            print(f"    ❌ {model_name} 评估失败: {e}")
            return {
                'model_name': model_name,
                'cv_scores': None,
                'cv_mean': -np.inf,
                'cv_std': 0,
                'model': model,
                'status': 'failed',
                'error': str(e)
            }
    
    def optimize_model(self, model, X, y, model_name):
        """优化模型超参数"""
        try:
            # 为不同模型设置不同的优化策略
            if model_name in ["LightGBM", "CatBoost"]:
                # 使用SSA优化
                ssa_params = {
                    "use_custom_ssa": True,
                    "param_bounds": {
                        "learning_rate": [0.01, 0.3],
                        "num_leaves": [10, 100]
                    },
                    "pop_size": 10,
                    "max_iter": 20
                }
                optimized_model, score, params = run_ssa_search(
                    model, X, y, ssa_params, self.cv, self.task_type, self.n_jobs
                )
                return optimized_model, score, params
            else:
                # 使用网格搜索
                if model_name == "RandomForest":
                    grid_params = {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [5, 10, 15]
                    }
                elif model_name == "SVM":
                    grid_params = {
                        "C": [0.1, 1, 10],
                        "gamma": ["scale", "auto"]
                    }
                else:
                    grid_params = {"n_estimators": [50, 100, 200]}
                
                optimized_model, score, params = run_grid_search(
                    model, X, y, grid_params, self.cv, self.task_type, self.n_jobs, self.random_state
                )
                return optimized_model, score, params
        except Exception as e:
            print(f"优化 {model_name} 时出错: {e}")
            return model, -np.inf, {}
    
    def auto_predict(self, X, y, optimize=True, preprocess=True):
        """执行自动化预测"""
        print("🚀 开始自动化基因组预测...")
        print(f"数据形状: {X.shape}, 任务类型: {self.task_type}")
        
        # 获取候选模型
        candidate_models = self.get_candidate_models()
        preprocessing_options = self.get_preprocessing_options(X.shape[1]) if preprocess else {"none": None}
        
        all_results = []
        
        # 遍历预处理选项
        for prep_name, prep_config in preprocessing_options.items():
            print(f"\n📊 测试预处理: {prep_name}")
            
            # 应用预处理
            if prep_config is None:
                X_processed = X
            else:
                X_processed = reduce_dimensions(
                    X, prep_config["method"], prep_config["n_components"]
                )
                print(f"降维后形状: {X_processed.shape}")
            
            # 遍历模型
            for model_name, model_info in candidate_models.items():
                print(f"  🔍 测试模型: {model_name}")
                
                try:
                    # 创建模型
                    model = get_model(
                        self.task_type, model_info["class"], 
                        gpu=False, categorical=False
                    )
                    
                    # 评估基础模型
                    base_result = self.evaluate_model(model, X_processed, y, model_name)
                    
                    if base_result['status'] == 'success':
                        result = {
                            'preprocessing': prep_name,
                            'model_name': model_name,
                            'base_cv_mean': base_result['cv_mean'],
                            'base_cv_std': base_result['cv_std'],
                            'optimized_cv_mean': base_result['cv_mean'],
                            'optimized_cv_std': base_result['cv_std'],
                            'model': base_result['model'],
                            'params': {},
                            'optimization': 'none'
                        }
                        
                        # 优化模型（如果启用）
                        if optimize and model_name in ["LightGBM", "CatBoost", "RandomForest", "SVM"]:
                            print(f"    ⚡ 优化 {model_name}...")
                            optimized_model, opt_score, opt_params = self.optimize_model(
                                model, X_processed, y, model_name
                            )
                            
                            if opt_score > base_result['cv_mean']:
                                result['model'] = optimized_model
                                result['optimized_cv_mean'] = opt_score
                                result['params'] = opt_params
                                result['optimization'] = 'ssa' if model_name in ["LightGBM", "CatBoost"] else 'grid'
                                print(f"    ✅ 优化成功: {opt_score:.6f}")
                            else:
                                print(f"    ⚠️  优化未改善性能")
                        
                        all_results.append(result)
                        print(f"    📈 CV得分: {result['optimized_cv_mean']:.6f} ± {result['optimized_cv_std']:.6f}")
                    
                except Exception as e:
                    print(f"    ❌ {model_name} 失败: {e}")
                    continue
        
        # 排序并选择最佳结果
        all_results.sort(key=lambda x: x['optimized_cv_mean'], reverse=True)
        
        self.results = {
            'all_results': all_results,
            'best_result': all_results[0] if all_results else None,
            'task_type': self.task_type,
            'cv': self.cv,
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        }
        
        return self.results
    
    def print_summary(self):
        """打印结果摘要"""
        if not self.results or not self.results['all_results']:
            print("❌ 没有可用的结果")
            return
        
        print("\n" + "="*60)
        print("🎯 自动化基因组预测结果摘要")
        print("="*60)
        
        best = self.results['best_result']
        print(f"🏆 最佳模型: {best['model_name']}")
        print(f"📊 预处理: {best['preprocessing']}")
        print(f"⚡ 优化方法: {best['optimization']}")
        if self.task_type == "regression":
            print(f"📈 皮尔逊相关系数: {best['optimized_cv_mean']:.6f} ± {best['optimized_cv_std']:.6f}")
        else:
            print(f"📈 准确率: {best['optimized_cv_mean']:.6f} ± {best['optimized_cv_std']:.6f}")
        if best['params']:
            print(f"🔧 最优参数: {best['params']}")
        
        print(f"\n📋 所有模型排名 (前10名):")
        print("-" * 60)
        if self.task_type == "regression":
            print(f"{'排名':<4} {'模型':<15} {'预处理':<12} {'皮尔逊相关系数':<15} {'优化':<8}")
        else:
            print(f"{'排名':<4} {'模型':<15} {'预处理':<12} {'准确率':<12} {'优化':<8}")
        print("-" * 60)
        
        for i, result in enumerate(self.results['all_results'][:10], 1):
            print(f"{i:<4} {result['model_name']:<15} {result['preprocessing']:<12} "
                  f"{result['optimized_cv_mean']:.6f} {result['optimization']:<8}")
    
    def save_detailed_results(self, output_file="detailed_results.txt"):
        """保存详细结果到文本文件"""
        if not self.results or not self.results['all_results']:
            print("❌ 没有可保存的结果")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("AIGP 自动化基因组预测详细结果\n")
            f.write("="*50 + "\n\n")
            
            # 基本信息
            f.write(f"任务类型: {self.results['task_type']}\n")
            f.write(f"交叉验证折数: {self.results['cv']}\n")
            f.write(f"样本数: {self.results['n_samples']}\n")
            f.write(f"特征数: {self.results['n_features']}\n\n")
            
            # 最佳结果
            if self.results.get('best_result'):
                best = self.results['best_result']
                f.write("最佳模型结果:\n")
                f.write("-" * 30 + "\n")
                f.write(f"模型名称: {best['model_name']}\n")
                f.write(f"预处理方法: {best['preprocessing']}\n")
                f.write(f"优化方法: {best['optimization']}\n")
                if self.task_type == "regression":
                    f.write(f"皮尔逊相关系数: {best['optimized_cv_mean']:.6f}\n")
                    f.write(f"标准差: {best['optimized_cv_std']:.6f}\n")
                else:
                    f.write(f"准确率: {best['optimized_cv_mean']:.6f}\n")
                    f.write(f"标准差: {best['optimized_cv_std']:.6f}\n")
                if best['params']:
                    f.write(f"最优参数: {best['params']}\n")
                f.write("\n")
            
            # 所有结果
            f.write("所有模型结果:\n")
            f.write("-" * 30 + "\n")
            if self.task_type == "regression":
                f.write(f"{'排名':<4} {'模型':<15} {'预处理':<12} {'皮尔逊相关系数':<15} {'标准差':<12} {'优化':<8}\n")
                f.write("-" * 80 + "\n")
                
                for i, result in enumerate(self.results['all_results'], 1):
                    f.write(f"{i:<4} {result['model_name']:<15} {result['preprocessing']:<12} "
                           f"{result['optimized_cv_mean']:<15.6f} {result['optimized_cv_std']:<12.6f} "
                           f"{result['optimization']:<8}\n")
                
                f.write("\n")
                f.write("注意: 使用皮尔逊相关系数评估模型性能\n")
            else:
                f.write(f"{'排名':<4} {'模型':<15} {'预处理':<12} {'准确率':<12} {'标准差':<12} {'优化':<8}\n")
                f.write("-" * 80 + "\n")
                
                for i, result in enumerate(self.results['all_results'], 1):
                    f.write(f"{i:<4} {result['model_name']:<15} {result['preprocessing']:<12} "
                           f"{result['optimized_cv_mean']:<12.6f} {result['optimized_cv_std']:<12.6f} "
                           f"{result['optimization']:<8}\n")
                
                f.write("\n")
                f.write("注意: 使用准确率评估模型性能\n")
        
        print(f"📁 详细结果已保存到: {output_file}")
    
    def get_best_model(self):
        """获取最佳模型"""
        if self.results and self.results['best_result']:
            return self.results['best_result']['model']
        return None
    
    def save_results(self, output_file="auto_predict_results.json"):
        """保存结果到文件"""
        import json
        
        if not self.results or not self.results.get('all_results'):
            print("❌ 没有可保存的结果")
            return
        
        # 准备可序列化的结果
        serializable_results = {
            'task_type': self.results['task_type'],
            'cv': self.results['cv'],
            'n_samples': self.results['n_samples'],
            'n_features': self.results['n_features'],
            'best_result': None,
            'all_results': [
                {
                    'model_name': r['model_name'],
                    'preprocessing': r['preprocessing'],
                    'cv_mean': r['optimized_cv_mean'],
                    'cv_std': r['optimized_cv_std'],
                    'optimization': r['optimization']
                } for r in self.results['all_results']
            ]
        }
        
        # 如果有最佳结果，添加最佳结果信息
        if self.results.get('best_result'):
            serializable_results['best_result'] = {
                'model_name': self.results['best_result']['model_name'],
                'preprocessing': self.results['best_result']['preprocessing'],
                'cv_mean': self.results['best_result']['optimized_cv_mean'],
                'cv_std': self.results['best_result']['optimized_cv_std'],
                'params': self.results['best_result']['params'],
                'optimization': self.results['best_result']['optimization']
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 结果已保存到: {output_file}")


def auto_predict(geno_file, phe_file, task_type="regression", cv=5, n_jobs=4, 
                phe_col_num=3, category_cols=None, optimize=True, preprocess=True):
    """
    一键式自动化基因组预测
    
    参数:
        geno_file: 基因型文件路径
        phe_file: 表型文件路径
        task_type: 任务类型 ("regression" 或 "classification")
        cv: 交叉验证折数
        n_jobs: 并行作业数
        phe_col_num: 表型列号
        category_cols: 协变量列号
        optimize: 是否进行超参数优化
        preprocess: 是否进行数据预处理
    """
    from .data_loader import load_training_data
    
    # 加载数据
    print("📂 加载数据...")
    X, y, covariates = load_training_data(
        geno_file, '\s+', phe_file, '\s+', 
        phe_col_num, category_cols, task_type
    )
    
    # 创建自动化预测器
    predictor = AutoGenomicPredictor(task_type=task_type, cv=cv, n_jobs=n_jobs)
    
    # 执行自动化预测
    results = predictor.auto_predict(X, y, optimize=optimize, preprocess=preprocess)
    
    # 打印结果
    predictor.print_summary()
    
    # 保存结果
    predictor.save_results()
    predictor.save_detailed_results()
    
    return predictor
