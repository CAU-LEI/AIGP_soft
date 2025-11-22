# AIGP Documentation Overview

## 📚 Documentation Structure

This documentation contains the following 4 parts to help you quickly master the use of the AIGP tool:

### 1. 📖 [AIGP_Data_Format_and_Index_Guide.md](AIGP_Data_Format_and_Index_Guide.md)
**Detailed Technical Documentation**
- Detailed explanation of genotypic and phenotypic data formats
- Index calculation rules and conversion formulas
- Data loading process description
- Debugging tools and validation methods
- Common errors and solutions

### 2. 🚀 [AIGP_Quick_Reference_Card.md](AIGP_Quick_Reference_Card.md)
**Quick Reference Card**
- Basic command format
- Common parameter descriptions
- Quick command examples
- Common error quick reference table

### 3. 🐎 [AIGP_Real_Case_Study.md](AIGP_Real_Case_Study.md)
**Case Study Based on Real Data**
- Horse genomic phenotype prediction case
- Detailed index calculation process
- Complete command examples
- Result analysis and performance comparison

### 4. 📋 [AIGP_DOCUMENTATION_OVERVIEW.md](AIGP_DOCUMENTATION_OVERVIEW.md)
**This Document - Overview and Navigation**

## 🎯 Quick Start

### If You Are a New User
1. First read [AIGP_Data_Format_and_Index_Guide.md](AIGP_Data_Format_and_Index_Guide.md) to understand basic concepts
2. Refer to [AIGP_Real_Case_Study.md](AIGP_Real_Case_Study.md) to learn specific operations
3. Use [AIGP_Quick_Reference_Card.md](AIGP_Quick_Reference_Card.md) for daily queries

### If You Are an Experienced User
1. Directly view [AIGP_Quick_Reference_Card.md](AIGP_Quick_Reference_Card.md) for command formats
2. Refer to [AIGP_Real_Case_Study.md](AIGP_Real_Case_Study.md) for best practices

## 🔑 Key Points

### Data Format Requirements
- **Genotypic Data**: Supports PLINK RAW, text, PED, VCF formats
- **Phenotypic Data**: Standard format, first column is sample ID
- **Classification Tasks**: Labels must start from 0 (0,1,2,3...)

### Index Calculation Rules
```
Actual Index = Original Column Position
```

**Important**: After data loading, the first column will be set as index, and subsequent column indices will be adjusted accordingly

### Common Command Format
```bash
python main.py \
  --geno <genotype_file> \
  --phe <phenotype_file> \
  --phe_sep " " \
  --phe_col_num <phenotype_column_index> \
  --type <task_type> \
  --model <model_name> \
  --cv <cross_validation_folds>
```

## 📊 Supported Features

### Machine Learning Algorithms
- **Regression Tasks**: XGBoost, LightGBM, CatBoost, RandomForest, SVM, etc.
- **Classification Tasks**: XGBoost, LightGBM, CatBoost, RandomForest, LogisticRegression, etc.

### Dimensionality Reduction Methods
- **PCA**: Principal Component Analysis
- **PHATE**: Potential of Heat-diffusion for Affinity-based Trajectory Embedding

### Optimization Methods
- **Grid Search**: Grid Search
- **Sparrow Search Algorithm**: Sparrow Search Algorithm

### Analysis Features
- **Cross-Validation**: K-fold cross-validation
- **SHAP Analysis**: Model interpretability analysis
- **Feature Importance**: Feature selection and analysis

## 🛠️ Debugging Tools

### Python Debugging Script
```python
import pandas as pd

def check_phenotype_index(file_path, sep=' '):
    """Automatically check phenotype file index"""
    df = pd.read_csv(file_path, sep=sep, header=0)
    df_indexed = df.set_index(df.columns[0])
    
    if 'PHENOTYPE' in df_indexed.columns:
        index = df_indexed.columns.get_loc('PHENOTYPE')
        print(f"Use command: --phe_col_num {index}")
    else:
        print("PHENOTYPE column not found")

# Usage example
check_phenotype_index('your_phenotype_file.txt', ' ')
```

### Command Line Check
```bash
# Check file structure
head -3 your_file.txt
awk '{print NF; exit}' your_file.txt
```

## ⚠️ Common Issues

### 1. Index Error
**Error**: `IndexError: single positional indexer is out-of-bounds`
**Solution**: Use debugging script to check actual index

### 2. Classification Label Error
**Error**: `ValueError: Invalid classes inferred from unique values`
**Solution**: Ensure classification labels start from 0

### 3. File Format Error
**Error**: `ValueError: Unknown genotype file format`
**Solution**: Use supported formats (.raw, .txt, .ped, .vcf)

## 📈 Performance Benchmarks

### Horse Genomic Prediction Case
| Method | Average Accuracy | Std Dev | Performance |
|--------|------------------|---------|-------------|
| PCA + XGBoost | 66.25% | 2.60% | Good |
| PHATE + XGBoost | 62.08% | 4.15% | Fair |

**Note**: Random guessing accuracy is 25% (4-class task), both methods significantly outperform random level.

## 🔄 Update Log

### v2.0.0 (Current Version)
- ✅ Redesigned documentation structure
- ✅ Added detailed index calculation instructions
- ✅ Provided real case demonstrations
- ✅ Improved debugging tools and validation methods
- ✅ Optimized quick reference card

### v1.0.0
- ✅ Initial version release
- ✅ Support for multiple machine learning algorithms
- ✅ Integrated dimensionality reduction and optimization features

## 📞 Technical Support

If you encounter problems during use:

1. **Consult Documentation**: First check relevant documentation
2. **Use Debugging Tools**: Run provided debugging scripts
3. **Check Common Issues**: Refer to common issue solutions
4. **Submit Issue**: Submit problem report in project repository

## 📄 License

This tool is for academic research use only. Please ensure compliance with relevant data usage agreements and laws and regulations.

---

💡 **Tip**: It is recommended to bookmark [AIGP_Quick_Reference_Card.md](AIGP_Quick_Reference_Card.md) for quick reference to common commands and parameters.

