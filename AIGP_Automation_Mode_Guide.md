# AIGP Automation Mode Detailed Guide

## 📋 Overview

The `--auto` flag enables AIGP's automated genomic prediction mode, which systematically tests multiple model–preprocessing combinations and automatically selects the best performing configuration. This document provides detailed specifications of all rules, parameters, workflows, and optimization strategies used in automation mode.

---

# 🚀 Command Usage

## Basic Command Format

```bash
python main.py --geno <genotype_file> --phe <phenotype_file> --type <task_type> --auto [options]
```

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|--------|----------|-------------------------------|
| `--auto` | flag | False | Enable automated prediction mode |
| `--auto_optimize` | flag | True | Enable hyperparameter optimization |
| `--auto_preprocess` | flag | True | Enable automatic preprocessing selection |
| `--cv` | int | 5 | Number of cross-validation folds |
| `--n_jobs` | int | 1 | Number of parallel jobs |
| `--type` | str | required | Task type: `regression` or `classification` |

## Example Commands

```bash
python main.py --geno data.raw --phe phenotype.txt --type regression --auto
python main.py --geno data.raw --phe phenotype.txt --type classification --auto --cv 10 --n_jobs 4
python main.py --geno data.raw --phe phenotype.txt --type regression --auto --auto_optimize False
python main.py --geno data.raw --phe phenotype.txt --type classification --auto --auto_preprocess False
```

---

# 🔁 Cross-Validation

- **Method**: KFold (default: 5)
- **Classification**: StratifiedKFold
- **Regression**: KFold
- **Random State**: 42
- **Parallelization**: via `--n_jobs`

---

# 🔄 Automation Workflow

## Step-by-Step Process

### 1. Data Loading
- Load genotype and phenotype data  
- Align samples by ID  
- Clean data based on task type  

### 2. Preprocessing Selection
- Analyze feature count  
- Choose preprocessing options  
- Build preprocessing configurations  

### 3. Model Testing Loop

```
For each preprocessing option:
    Apply preprocessing
    For each model:
        Initialize model
        Evaluate base model (CV)
        If optimization enabled:
            Run optimization (SSA or Grid)
            Compare optimized vs base
        Save result
```

### 4. Result Ranking
- Sort by `optimized_cv_mean`  
- Select best configuration  

### 5. Output Generation
- Console summary  
- JSON results  
- Text report  

---

# 📁 Output Files

## JSON Results Example

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
  }
}
```

---

# 🎯 Best Practices

## 1. Start with Defaults
```bash
python main.py --geno data.raw --phe phe.txt --type regression --auto
```

## 2. Large Datasets
- Use 4+ threads  
- Disable preprocessing if features < 1000  
- Lower CV folds to 3  

## 3. Quick Testing
- Disable optimization + preprocessing  

## 4. Best Accuracy
- Keep all automation enabled  
- 5–10 CV folds recommended  

---

# 💡 Tip

All implementation details can be found in:

```
aigp/auto_predictor.py
```
