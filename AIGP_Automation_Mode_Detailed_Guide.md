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
'''


**Optimization Process**:
1. Generate all parameter combinations
2. Evaluate each combination using cross-validation
3. Select combination with best performance
4. Return optimized model


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



## 📁 Output Files

### 1. Console Output
- Real-time progress for each model-preprocessing combination
- Performance metrics for each evaluation
- Optimization status and results
- Final summary with top 10 models

### 2. JSON Results File ()
```json
{
  "task_type": "regression",
  "cv": 5,
  "n_samples": 480,
  "n_features": 50621,
  "best_result": {
    "model_name": "XGBoost",
    "preprocessing": "none",
    "cv_mean": 0.86,
    "cv_std": 0.02,
    "params": {},
    "optimization": "none"
  },
  "all_results": [
    {
      "model_name": "XGBoost",
      "preprocessing": "none",
      "cv_mean": 0.86,
      "cv_std": 0.02,
      "optimization": "none"
    },
    ...
  ]
}
```

### 3. Detailed Text Report ()
- Task information
- Best model details
- Complete ranking of all models
- Performance statistics




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




---

💡 **Tip**: For detailed technical implementation, refer to the source code in `aigp/auto_predictor.py`.

