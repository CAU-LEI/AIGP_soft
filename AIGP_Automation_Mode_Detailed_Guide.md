# AIGP Automation Mode Detailed Guide

## 📋 Overview

The `--auto` flag enables AIGP's automated genomic prediction mode, which systematically tests multiple model-preprocessing combinations and automatically selects the best performing configuration. This document provides detailed specifications of all rules, parameters, and optimization strategies used in automation mode.

## 🚀 Command Usage

### Basic Command Format
```bash
python main.py --geno <genotype_file> --phe <phenotype_file> --type <task_type> --auto [options]
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--auto` | flag | False | Enable automated prediction mode |
| `--auto_optimize` | flag | True | Enable hyperparameter optimization |
| `--auto_preprocess` | flag | True | Enable automatic preprocessing selection |
| `--cv` | int | 5 | Number of cross-validation folds |
| `--n_jobs` | int | 1 | Number of parallel jobs |
| `--type` | str | required | Task type: `regression` or `classification` |

### Example Commands
```bash
# Basic automation (all defaults)
python main.py --geno data.raw --phe phenotype.txt --type regression --auto

# Custom cross-validation and parallel jobs
python main.py --geno data.raw --phe phenotype.txt --type classification --auto --cv 10 --n_jobs 4

# Disable optimization (faster, but less accurate)
python main.py --geno data.raw --phe phenotype.txt --type regression --auto --auto_optimize False

# Disable preprocessing (use raw features only)
python main.py --geno data.raw --phe phenotype.txt --type classification --auto --auto_preprocess False
```

## 📊 Preprocessing Options (Dynamic Selection)

The system automatically selects preprocessing options based on the number of features in your data:

### Selection Rules

| Feature Count | Preprocessing Options | Description |
|---------------|----------------------|-------------|
| Any | `none` | No preprocessing (always included) |
| > 100 | `phate_50` | PHATE dimensionality reduction to 50 components |
| > 500 | `pca_50` | PCA dimensionality reduction to 50 components |
| > 1000 | `pca_100`, `pca_200` | PCA dimensionality reduction to 100 or 200 components |

### Detailed Preprocessing Configurations

#### 1. None (No Preprocessing)
- **Name**: `none`
- **Method**: None
- **Output**: Original feature matrix
- **Use Case**: Always tested for baseline comparison

#### 2. PCA Options
- **pca_50**: Principal Component Analysis to 50 dimensions
  - **Condition**: n_features > 500
  - **Method**: `sklearn.decomposition.PCA`
  - **Components**: 50
  
- **pca_100**: Principal Component Analysis to 100 dimensions
  - **Condition**: n_features > 1000
  - **Method**: `sklearn.decomposition.PCA`
  - **Components**: 100
  
- **pca_200**: Principal Component Analysis to 200 dimensions
  - **Condition**: n_features > 1000
  - **Method**: `sklearn.decomposition.PCA`
  - **Components**: 200

#### 3. PHATE Option
- **phate_50**: PHATE dimensionality reduction to 50 dimensions
  - **Condition**: n_features > 100
  - **Method**: `phate.PHATE`
  - **Components**: 50

### Example: Preprocessing Selection

For a dataset with **50,621 features**:
- ✅ `none` (always included)
- ✅ `phate_50` (50,621 > 100)
- ✅ `pca_50` (50,621 > 500)
- ✅ `pca_100` (50,621 > 1000)
- ✅ `pca_200` (50,621 > 1000)
- **Total**: 5 preprocessing options

## 🤖 Model Lists

### Regression Tasks (11 Models)

| Model Name | Class Name | Priority | Optimization |
|------------|------------|----------|--------------|
| LightGBM | LGBMRegressor | 1 | SSA |
| CatBoost | CatBoostRegressor | 1 | SSA |
| XGBoost | xgboost | 2 | None |
| RandomForest | RandomForest | 2 | Grid Search |
| GradientBoosting | GradientBoosting | 3 | None |
| SVM | svm | 3 | Grid Search |
| KNN | knn | 4 | None |
| Ridge | RidgeRegression | 4 | None |
| LinearRegression | LinearRegression | 4 | None |
| ElasticNet | ElasticNet | 4 | None |
| AdaBoost | AdaBoost | 4 | None |

### Classification Tasks (10 Models)

| Model Name | Class Name | Priority | Optimization |
|------------|------------|----------|--------------|
| LightGBM | LGBM | 1 | SSA |
| CatBoost | CatBoost | 1 | SSA |
| XGBoost | xgboost | 2 | None |
| RandomForest | RandomForest | 2 | Grid Search |
| GradientBoosting | GradientBoosting | 3 | None |
| SVM | svm | 3 | Grid Search |
| KNN | knn | 4 | None |
| LogisticRegression | LogisticRegression | 4 | None |
| AdaBoost | AdaBoost | 4 | None |
| ExtraTrees | ExtraTrees | 4 | None |

## ⚙️ Hyperparameter Optimization

### Optimization Strategy

Optimization is only applied to **4 specific models**:
- **SSA Optimization**: LightGBM, CatBoost
- **Grid Search Optimization**: RandomForest, SVM
- **No Optimization**: All other models

### SSA (Sparrow Search Algorithm) Configuration

**Applied to**: LightGBM, CatBoost

**Parameters**:
```python
ssa_params = {
    "use_custom_ssa": True,
    "param_bounds": {
        "learning_rate": [0.01, 0.3],  # Search range
        "num_leaves": [10, 100]         # Search range
    },
    "pop_size": 10,      # Population size
    "max_iter": 20       # Maximum iterations
}
```

**Optimization Process**:
1. Initialize population of 10 candidate solutions
2. Run 20 iterations of SSA algorithm
3. Each iteration evaluates candidates using cross-validation
4. Returns best parameters found

**Search Space**:
- `learning_rate`: Continuous range [0.01, 0.3]
- `num_leaves`: Integer range [10, 100]

### Grid Search Configuration

**Applied to**: RandomForest, SVM

#### RandomForest Grid Search
```python
grid_params = {
    "n_estimators": [50, 100, 200],    # 3 values
    "max_depth": [5, 10, 15]            # 3 values
}
# Total combinations: 3 × 3 = 9 combinations
```

#### SVM Grid Search
```python
grid_params = {
    "C": [0.1, 1, 10],                  # 3 values
    "gamma": ["scale", "auto"]          # 2 values
}
# Total combinations: 3 × 2 = 6 combinations
```

**Optimization Process**:
1. Generate all parameter combinations
2. Evaluate each combination using cross-validation
3. Select combination with best performance
4. Return optimized model

## 📈 Evaluation Metrics

### Regression Tasks
- **Metric**: Pearson Correlation Coefficient
- **Scorer**: Custom scorer using `scipy.stats.pearsonr`
- **Range**: -1 to 1 (higher is better)
- **Calculation**: 
  ```python
  def pearson_corr(y_true, y_pred):
      if len(y_true) < 2:
          return 0
      corr, _ = pearsonr(y_true, y_pred)
      return corr
  ```

### Classification Tasks
- **Metric**: Accuracy
- **Scorer**: `sklearn.metrics.accuracy_score`
- **Range**: 0 to 1 (higher is better)

### Cross-Validation
- **Method**: K-Fold Cross-Validation (default: 5-fold)
- **Stratification**: StratifiedKFold for classification, KFold for regression
- **Random State**: 42 (fixed for reproducibility)
- **Parallelization**: Uses `n_jobs` parameter for parallel execution

## 🔄 Automation Workflow

### Step-by-Step Process

1. **Data Loading**
   - Load genotype and phenotype data
   - Align samples by ID
   - Clean data based on task type

2. **Preprocessing Selection**
   - Analyze feature count
   - Generate preprocessing options based on rules
   - Create preprocessing configurations

3. **Model Testing Loop**
   ```
   For each preprocessing option:
       Apply preprocessing to data
       For each model:
           Create model instance
           Evaluate base model (cross-validation)
           If optimization enabled and model in [LightGBM, CatBoost, RandomForest, SVM]:
               Run hyperparameter optimization
               Compare optimized vs base performance
               Use better performing version
           Store results
   ```

4. **Result Ranking**
   - Sort all results by `optimized_cv_mean` (descending)
   - Select best result
   - Generate summary report

5. **Output Generation**
   - Print summary to console
   - Save JSON results file
   - Save detailed text report

## 📊 Combination Calculation

### Example: Regression Task with 50,621 Features

**Preprocessing Options**: 5 (none, phate_50, pca_50, pca_100, pca_200)
**Models**: 11
**Total Combinations**: 5 × 11 = **55 combinations**

**Optimization Breakdown**:
- **SSA Optimization**: 5 preprocessing × 2 models (LightGBM, CatBoost) = **10 combinations**
- **Grid Search Optimization**: 5 preprocessing × 2 models (RandomForest, SVM) = **10 combinations**
- **No Optimization**: 5 preprocessing × 7 models = **35 combinations**

### Example: Classification Task with 1,500 Features

**Preprocessing Options**: 4 (none, phate_50, pca_50, pca_100, pca_200)
**Models**: 10
**Total Combinations**: 4 × 10 = **40 combinations**

**Optimization Breakdown**:
- **SSA Optimization**: 4 preprocessing × 2 models = **8 combinations**
- **Grid Search Optimization**: 4 preprocessing × 2 models = **8 combinations**
- **No Optimization**: 4 preprocessing × 6 models = **24 combinations**

## 📁 Output Files

### 1. Console Output
- Real-time progress for each model-preprocessing combination
- Performance metrics for each evaluation
- Optimization status and results
- Final summary with top 10 models

### 2. JSON Results File (`auto_predict_results.json`)
```json
{
  "task_type": "regression",
  "cv": 5,
  "n_samples": 480,
  "n_features": 50621,
  "best_result": {
    "model_name": "XGBoost",
    "preprocessing": "none",
    "cv_mean": 0.8625,
    "cv_std": 0.0212,
    "params": {},
    "optimization": "none"
  },
  "all_results": [
    {
      "model_name": "XGBoost",
      "preprocessing": "none",
      "cv_mean": 0.8625,
      "cv_std": 0.0212,
      "optimization": "none"
    },
    ...
  ]
}
```

### 3. Detailed Text Report (`detailed_results.txt`)
- Task information
- Best model details
- Complete ranking of all models
- Performance statistics

## ⚡ Performance Considerations

### Time Estimation

**Base Model Evaluation**:
- Each combination: ~1-5 minutes (depending on data size and CV folds)
- 55 combinations: ~55-275 minutes (without optimization)

**With Optimization**:
- SSA optimization: +5-15 minutes per combination
- Grid Search: +2-10 minutes per combination
- Total with optimization: ~2-8 hours (depending on data size)

### Parallelization

- Use `--n_jobs` to enable parallel processing
- Recommended: `--n_jobs 4` or `--n_jobs 8`
- Speeds up cross-validation and grid search
- SSA optimization runs sequentially

## 🎯 Best Practices

### 1. Start with Defaults
```bash
python main.py --geno data.raw --phe phe.txt --type regression --auto
```

### 2. For Large Datasets
- Use `--n_jobs 4` or higher for parallelization
- Consider disabling preprocessing if features < 1000
- Use fewer CV folds (e.g., `--cv 3`) for faster results

### 3. For Quick Testing
- Disable optimization: `--auto_optimize False`
- Disable preprocessing: `--auto_preprocess False`
- Use fewer CV folds: `--cv 3`

### 4. For Best Results
- Use default settings (optimization and preprocessing enabled)
- Use 5-10 CV folds for reliable estimates
- Allow sufficient time for completion

## 🔍 Understanding Results

### Result Fields

- **preprocessing**: Preprocessing method used
- **model_name**: Machine learning model name
- **base_cv_mean**: Base model performance (before optimization)
- **optimized_cv_mean**: Optimized model performance (after optimization)
- **optimization**: Optimization method used (`ssa`, `grid`, or `none`)
- **params**: Optimal hyperparameters (if optimization was applied)

### Performance Comparison

The system automatically compares:
- Base model vs optimized model
- Uses optimized version if it performs better
- Falls back to base model if optimization doesn't improve performance



---

💡 **Tip**: For detailed technical implementation, refer to the source code in `aigp/auto_predictor.py`.

