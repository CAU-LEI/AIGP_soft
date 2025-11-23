# AIGP Automated Genomic Prediction 

## Overview

AIGP now integrates a comprehensive **Automated Genomic Prediction
Module**, fundamentally transforming the conventional workflow in
genomic data analysis.\
This innovative module fully automates model selection, hyperparameter
tuning, preprocessing strategies, and performance benchmarking ---
enabling researchers to focus on scientific insights rather than
technical implementation.

## Key Advantages
### 1.Adaptive data preprocessing
The system automatically detects data characteristics, selects appropriate dimensionality reduction methods (PCA or PHATE), and supports mainstream input formats such as PLINK, PED, and VCF. It performs comprehensive missing-value checks, data alignment, and multi-format compatibility, thereby minimizing manual preprocessing and format conversion efforts.
### 2.Automated model testing and selection
AIGP now supports parallel testing of more than ten mainstream machine learning algorithms. It automatically performs dimensionality reduction when appropriate, matches regression or classification models according to the task type, compares all model performances in detail, and identifies the optimal model. This automation greatly reduces computation time while ensuring consistent and reproducible results.
### 3.Hardware detection and adaptive acceleration
The new hardware detection and optimization module further enhances automation. Upon startup, AIGP automatically identifies the operating system and hardware configuration (CPU type, GPU availability, memory size, and core count) and selects the optimal computational strategy accordingly. When GPU resources (e.g., NVIDIA CUDA) are available, the system switches automatically to GPU mode for substantial speed-up. Otherwise, it enables multi-core parallel computation, dynamically adjusting the number of threads based on CPU and memory capacity. Thread affinity and memory prefetch parameters are automatically configured to reduce I/O latency and ensure stable performance across platforms.




## How to Use

### Basic Command

``` bash
python main.py --geno <genotype_file> --phe <phenotype_file> --type <task_type> --auto
```

### Examples

``` bash
# Horse genomic classification (4-class task)
python main.py --geno horse.raw --phe horse_phe.txt --type classification --auto --cv 10

# Regression task
python main.py --geno data.raw --phe phenotype.txt --type regression --auto --cv 5
```

## Analysis Outputs

### Real-Time Analysis Display

-   Progress tracking for each model and preprocessing pipeline
-   Real-time performance feedback
-   Hyperparameter optimization progress and results

### Final Outputs

-   **Performance ranking table**
-   **Best model recommendation**
-   **JSON result file**
-   **Full text report**



### Automated Pipeline

1.  Data loading
2.  Preprocessing
3.  Model evaluation
4.  Hyperparameter tuning
5.  Result ranking

### Supported Algorithms

-   Regression: XGBoost, LightGBM, CatBoost, RF, SVM, KNN, Ridge,
    ElasticNet, etc.
-   Classification: XGBoost, LightGBM, CatBoost, RF, SVM, KNN, Logistic
    Regression, etc.



## Conclusion

The automated genomic prediction module in AIGP marks a major step
forward in genomic data analysis. It enables researchers to obtain the
best model with a single command, reduces days of tuning to minutes,
guarantees systematic evaluation, and shifts the focus back to
biological discovery.

*This document introduces the automated genomic prediction module of
AIGP v2.0.*
