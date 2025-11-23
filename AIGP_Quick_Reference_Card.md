# AIGP Quick Reference Card

## 🚀 Basic Command Format
```bash
python main.py --geno <genotype_file> --phe <phenotype_file> --type <task_type> [other_parameters]
```

## 📊 Data Format and Index

### Genotype Data (.raw format)
```
FID IID PAT MAT SEX PHENOTYPE SNP1 SNP2 SNP3 ...
1   1   0   0   1   2         1    0    2    ...
```
- **Genotype Features**: Starting from column 7 (0-based index)
- **Automatic Processing**: System automatically extracts all SNP columns

### Phenotype Data Format
```
FID IID PHENOTYPE
1   1   3.45
2   2   2.18
```

### Index Calculation Rules
```
Actual Index = Original Column Position
```

**Important**: The first column is the default index, and the position of subsequent columns is the actual index!

| Original Column Position | Actual Index | Command Parameter |
|--------------------------|--------------|-------------------|
| Column 2 | 2 | `--phe_col_num 2` |
| Column 3 | 3 | `--phe_col_num 3` |
| Column 4 | 4 | `--phe_col_num 4` |
| Column 5 | 5 | `--phe_col_num 5` |

## 🎯 Common Commands

### Basic Regression
```bash
python main.py --geno data.raw --phe phe.txt --phe_sep " " --phe_col_num 3 --type regression --model xgboost
```

### Classification + Cross-Validation
```bash
python main.py --geno data.raw --phe phe.txt --phe_sep " " --phe_col_num 3 --type classification --model xgboost --cv 10
```

### PCA Dimensionality Reduction
```bash
python main.py --geno data.raw --phe phe.txt --phe_sep " " --phe_col_num 3 --type classification --model xgboost --dim_reduction pca --n_components 100 --cv 10
```

### PHATE Dimensionality Reduction
```bash
python main.py --geno data.raw --phe phe.txt --phe_sep " " --phe_col_num 3 --type classification --model xgboost --dim_reduction phate --n_components 100 --cv 10
```

## 🔧 Debugging Tools

### Python Debugging Script
```python
import pandas as pd

# Check phenotype file index
df = pd.read_csv('phenotype.txt', sep=' ', header=0)
df_indexed = df.set_index(df.columns[0])
phe_index = df_indexed.columns.get_loc('PHENOTYPE')
print(f"Use: --phe_col_num {phe_index}")
```

### Command Line Check
```bash
# Check file structure
head -3 your_file.txt
awk '{print NF; exit}' your_file.txt
```

## ⚠️ Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `IndexError: out-of-bounds` | Column index error | Check `--phe_col_num` |
| `Invalid classes` | Classification labels don't start from 0 | Convert labels to 0,1,2,3... |
| `Unknown file format` | Unsupported extension | Use .raw/.txt/.ped/.vcf |

## 📋 Supported File Formats

| Format | Extension | Genotype Columns | Phenotype Columns |
|--------|-----------|------------------|-------------------|
| PLINK RAW | .raw | Starting from column 7 | Column 6 |
| Text Format | .txt | Starting from column 2 | Need to specify |
| PLINK PED | .ped | Starting from column 7 | Column 6 |
| VCF Format | .vcf | Starting from column 10 | Need to specify |



