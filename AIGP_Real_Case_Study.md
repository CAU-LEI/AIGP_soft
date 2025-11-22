# AIGP Real Case Study

## 🐎 Case: Horse Genomic Phenotype Prediction

### Data File Description

#### 1. Genotype File: `horse_480_use.raw`
```
FID IID PAT MAT SEX PHENOTYPE SNP1 SNP2 SNP3 SNP4 ...
1   1   0   0   1   2         1    0    2    1    ...
2   2   0   0   2   1         0    1    1    2    ...
3   3   0   0   1   2         2    0    0    1    ...
```

**File Structure Analysis**:
- **Total Columns**: 482 columns (1 sample ID + 481 SNP features)
- **Genotype Data**: Starting from column 7 (index 5)
- **Number of Features**: 476 SNP features

#### 2. Phenotype File: `horse_phe_0indexed.txt`
```
FID IID PHENOTYPE
1   1   0
2   2   2
3   3   0
```

**File Structure Analysis**:
- **Total Columns**: 3 columns
- **Phenotype Data**: Column 3 (PHENOTYPE)
- **Classification Labels**: 0, 1, 2, 3 (4-class task)

### Index Calculation Process

#### Step 1: Determine Original Column Position
- PHENOTYPE is in column 3 of the original file
- Original column position = 3

#### Step 2: Use Original Position Directly
```
Actual Index = Original Column Position
Actual Index = 3
```

#### Step 3: Verify Data Loading
```python
import pandas as pd

# Read phenotype file
df = pd.read_csv('horse_phe_0indexed.txt', sep=' ', header=0)
print("Original columns:", list(df.columns))
# Output: ['FID', 'IID', 'PHENOTYPE']

# Set FID as index
df_indexed = df.set_index('FID')
print("After setting index:", list(df_indexed.columns))
# Output: ['IID', 'PHENOTYPE']

# Check PHENOTYPE column index
phe_index = df_indexed.columns.get_loc('PHENOTYPE')
print(f"PHENOTYPE column index: {phe_index}")
# Output: PHENOTYPE column index: 1
```

**Important Finding**: The actual index is 3, use the original column position directly!

#### Step 4: Confirm Index
- Original position: Column 3
- Actual index: 3 ✅
- Reason: First column is default index, subsequent column positions are actual indices

### Correct Commands

#### 1. Basic Classification Task
```bash
python main.py \
  --geno "horse_480_use.raw" \
  --phe "horse_phe_0indexed.txt" \
  --phe_sep " " \
  --phe_col_num 3 \
  --type classification \
  --model xgboost
```

#### 2. 10-Fold Cross-Validation
```bash
python main.py \
  --geno "horse_480_use.raw" \
  --phe "horse_phe_0indexed.txt" \
  --phe_sep " " \
  --phe_col_num 3 \
  --type classification \
  --model xgboost \
  --cv 10 \
  --result_file "horse_results.txt"
```

#### 3. PCA Dimensionality Reduction (100 dimensions)
```bash
python main.py \
  --geno "horse_480_use.raw" \
  --phe "horse_phe_0indexed.txt" \
  --phe_sep " " \
  --phe_col_num 3 \
  --type classification \
  --model xgboost \
  --dim_reduction pca \
  --n_components 100 \
  --cv 10 \
  --result_file "horse_pca_results.txt"
```

#### 4. PHATE Dimensionality Reduction (100 dimensions)
```bash
python main.py \
  --geno "horse_480_use.raw" \
  --phe "horse_phe_0indexed.txt" \
  --phe_sep " " \
  --phe_col_num 3 \
  --type classification \
  --model xgboost \
  --dim_reduction phate \
  --n_components 100 \
  --cv 10 \
  --result_file "horse_phate_results.txt"
```

### Result Analysis

#### PCA + XGBoost Results
```
=== 10-Fold Cross Validation Results ===
Fold  1: 0.6250
Fold  2: 0.6875
Fold  3: 0.7083
Fold  4: 0.6458
Fold  5: 0.6875
Fold  6: 0.6250
Fold  7: 0.6667
Fold  8: 0.6667
Fold  9: 0.6667
Fold 10: 0.6458
----------------------------------------
Mean:     0.6625
Std Dev:  0.0260
Min:      0.6250
Max:      0.7083
```

#### PHATE + XGBoost Results
```
=== 10-Fold Cross Validation Results ===
Fold  1: 0.5833
Fold  2: 0.6250
Fold  3: 0.5417
Fold  4: 0.6458
Fold  5: 0.6875
Fold  6: 0.6250
Fold  7: 0.6667
Fold  8: 0.6458
Fold  9: 0.5833
Fold 10: 0.6042
----------------------------------------
Mean:     0.6208
Std Dev:  0.0415
Min:      0.5417
Max:      0.6875
```

### Performance Comparison

| Method | Average Accuracy | Std Dev | Performance |
|--------|------------------|---------|-------------|
| PCA + XGBoost | 66.25% | 2.60% | Good |
| PHATE + XGBoost | 62.08% | 4.15% | Fair |

**Analysis**:
- PCA dimensionality reduction performs better
- Random guessing accuracy: 25% (4-class)
- Both methods significantly outperform random level

## 🔍 Debugging Tips

### 1. Automatic Index Check Script
```python
def check_phenotype_index(file_path, sep=' '):
    """Automatically check phenotype file index"""
    df = pd.read_csv(file_path, sep=sep, header=0)
    df_indexed = df.set_index(df.columns[0])
    
    print(f"File: {file_path}")
    print(f"Original columns: {list(df.columns)}")
    print(f"After setting index: {list(df_indexed.columns)}")
    
    if 'PHENOTYPE' in df_indexed.columns:
        index = df_indexed.columns.get_loc('PHENOTYPE')
        print(f"PHENOTYPE column index: {index}")
        print(f"Use command: --phe_col_num {index}")
    else:
        print("PHENOTYPE column not found")
        print("Available columns:", list(df_indexed.columns))

# Usage example
check_phenotype_index('horse_phe_0indexed.txt', ' ')
```

### 2. Data Quality Check
```python
def check_data_quality(geno_file, phe_file):
    """Check data quality"""
    # Check genotype data
    geno_df = pd.read_csv(geno_file, sep=r'\s+', header=0)
    geno_df = geno_df.set_index('IID')
    X = geno_df.iloc[:, 5:]
    
    print("Genotype Data:")
    print(f"  Number of samples: {len(X)}")
    print(f"  Number of features: {len(X.columns)}")
    print(f"  Missing values: {X.isnull().sum().sum()}")
    
    # Check phenotype data
    phe_df = pd.read_csv(phe_file, sep=' ', header=0)
    phe_df = phe_df.set_index('FID')
    y = phe_df.iloc[:, 1]
    
    print("Phenotype Data:")
    print(f"  Number of samples: {len(y)}")
    print(f"  Class distribution: {y.value_counts().sort_index()}")
    print(f"  Missing values: {y.isnull().sum()}")

# Usage example
check_data_quality('horse_480_use.raw', 'horse_phe_0indexed.txt')
```

## 📋 Common Problem Solutions

### Problem 1: Index Error
**Error Message**: `IndexError: single positional indexer is out-of-bounds`

**Solution Steps**:
1. Use debugging script to check actual index
2. Confirm conversion formula
3. Verify data loading process

### Problem 2: Classification Label Error
**Error Message**: `ValueError: Invalid classes inferred from unique values`

**Solution**:
```python
# Convert classification labels
df['PHENOTYPE'] = df['PHENOTYPE'] - 1  # 1,2,3,4 → 0,1,2,3
```

### Problem 3: File Format Error
**Error Message**: `ValueError: Unknown genotype file format`

**Solution**:
- Check file extension
- Use supported formats: .raw, .txt, .ped, .vcf

## 🎯 Best Practices Summary

### 1. Data Preparation
- Use PLINK RAW format
- Ensure sample IDs are consistent
- Check data quality

### 2. Index Calculation
- Use debugging script to verify
- Don't blindly apply formulas
- Check actual loading results

### 3. Command Testing
- Test with small dataset first
- Check output results
- Verify cross-validation results

### 4. Result Analysis
- Compare different methods
- Check performance stability
- Verify biological significance

---

💡 **Tip**: When encountering problems, please use the provided debugging scripts to check your data format and index settings.

