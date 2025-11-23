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

# AutoGenomicPredictor Output Format Specification

This document describes the output formats used by the `AutoGenomicPredictor` automated predictor — specifically the JSON result file (`auto_predict_results.json`) and the detailed report file (`detailed_results.txt`). These formats facilitate result interpretation, downstream integration, and reporting.

---

## 1. JSON Result File (`auto_predict_results.json`)

Saved using `predictor.save_results(output_file)`, with the default filename `auto_predict_results.json`.

### Top-Level Structure

```json
{
  "task_type": "regression",        // Task type: "regression" or "classification"
  "cv": 5,                          // Number of cross-validation folds (int)
  "n_samples": 420,                 // Number of samples (int)
  "n_features": 968,                // Number of features (int)
  "best_result": { ... },           // Best model result (see below)
  "all_results": [ ... ]            // All model results list (see below)
}
```

### Field Descriptions

#### best_result (Best Model Result)

- `model_name`     : Model name (e.g., "LightGBM", "RandomForest")
- `preprocessing`  : Preprocessing method (e.g., "none", "pca_100", "phate_50")
- `cv_mean`        : Mean cross-validation score  
    - For "regression": Pearson correlation coefficient; for "classification": accuracy
- `cv_std`         : Standard deviation of cross-validation scores
- `params`         : Optimized parameters (key-value dict; empty if none)
- `optimization`   : Optimization method ("ssa": SSA optimization, "grid": Grid search, "none": no optimization)

#### all_results (All Model Results List)

- Each element is a dict including:
    - `model_name`, `preprocessing`, `cv_mean`, `cv_std`, `optimization`
    - Note: does not include `params`

### Example Fragment

```json
{
  "task_type": "regression",
  "cv": 5,
  "n_samples": 420,
  "n_features": 968,
  "best_result": {
    "model_name": "LightGBM",
    "preprocessing": "none",
    "cv_mean": 0.853468,
    "cv_std": 0.034112,
    "params": { "learning_rate": 0.16, "num_leaves": 57 },
    "optimization": "ssa"
  },
  "all_results": [
    {
      "model_name": "LightGBM",
      "preprocessing": "none",
      "cv_mean": 0.853468,
      "cv_std": 0.034112,
      "optimization": "ssa"
    },
    {
      "model_name": "RandomForest",
      "preprocessing": "pca_50",
      "cv_mean": 0.781357,
      "cv_std": 0.041225,
      "optimization": "grid"
    }
    // ... Other model results
  ]
}
```

---

## 2. Report Result File (`detailed_results.txt`)

Saved using `predictor.save_detailed_results(output_file)`, default filename is `detailed_results.txt`.

### Basic Structure

- Task information (task type, CV folds, sample number, feature number) at the top
- Best model summary
- All model results in tabular format (top 10 shown in summary; full list in the file)
- "regression": Pearson correlation coefficient; "classification": accuracy
- Note about evaluation metric at the end

### Example Sections

```
AIGP Automated Genomic Prediction Detailed Results
==================================================

Task type: regression
Cross-validation folds: 5
Number of samples: 420
Number of features: 968

Best model result:
------------------------------
Model name: LightGBM
Preprocessing: none
Optimization method: ssa
Pearson correlation coefficient: 0.853468
Standard deviation: 0.034112
Best parameters: {'learning_rate': 0.16, 'num_leaves': 57}

All model results:
------------------------------
Rank Model name       Preprocessing  Pearson coeff.   Std dev      Optimization
---- ---------------  ------------  ---------------  ----------   ------------
1    LightGBM         none          0.853468         0.034112     ssa
2    RandomForest     pca_50        0.781357         0.041225     grid
...  ...              ...           ...              ...          ...

Note: Models are evaluated by the Pearson correlation coefficient.
```

**For classification tasks, display “Accuracy” in the respective fields and end with:**  
Note: Models are evaluated by accuracy.

---

## 3. Important Notes

- All numeric values are floats with six decimal digits.
- If no results are available, "best_result" will be null and the model list will be empty.
- If no hyperparameter optimization is performed, `params` is `{}` and `optimization` is "none".
- File encoding: UTF-8

---

## 4. Recommended Usage

- Use `auto_predict_results.json` for pipeline integration, front-end display, or further analysis.
- Use `detailed_results.txt` for human reading, archiving, or inclusion in reports.

---

# 💡 Tip

All implementation details can be found in:

```
aigp/auto_predictor.py
```
