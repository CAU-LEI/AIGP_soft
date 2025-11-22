# AIGP 快速参考卡

## 🚀 基本命令格式
```bash
python main.py --geno <基因型文件> --phe <表型文件> --type <任务类型> [其他参数]
```

## 📊 数据格式和索引

### 基因型数据（.raw格式）
```
FID IID PAT MAT SEX PHENOTYPE SNP1 SNP2 SNP3 ...
1   1   0   0   1   2         1    0    2    ...
```
- **基因型特征**：从第7列开始（索引0开始）
- **自动处理**：系统自动提取所有SNP列

### 表型数据格式
```
FID IID PHENOTYPE
1   1   3.45
2   2   2.18
```

### 索引计算规则
```
实际索引 = 原始列位置
```

**重要**：第一列是默认索引，后续列的位置就是实际索引！

| 原始列位置 | 实际索引 | 命令参数 |
|------------|----------|----------|
| 第2列 | 2 | `--phe_col_num 2` |
| 第3列 | 3 | `--phe_col_num 3` |
| 第4列 | 4 | `--phe_col_num 4` |
| 第5列 | 5 | `--phe_col_num 5` |

## 🎯 常用命令

### 基本回归
```bash
python main.py --geno data.raw --phe phe.txt --phe_sep " " --phe_col_num 3 --type regression --model xgboost
```

### 分类+交叉验证
```bash
python main.py --geno data.raw --phe phe.txt --phe_sep " " --phe_col_num 3 --type classification --model xgboost --cv 10
```

### PCA降维
```bash
python main.py --geno data.raw --phe phe.txt --phe_sep " " --phe_col_num 3 --type classification --model xgboost --dim_reduction pca --n_components 100 --cv 10
```

### PHATE降维
```bash
python main.py --geno data.raw --phe phe.txt --phe_sep " " --phe_col_num 3 --type classification --model xgboost --dim_reduction phate --n_components 100 --cv 10
```

## 🔧 调试工具

### Python调试脚本
```python
import pandas as pd

# 检查表型文件索引
df = pd.read_csv('phenotype.txt', sep=' ', header=0)
df_indexed = df.set_index(df.columns[0])
phe_index = df_indexed.columns.get_loc('PHENOTYPE')
print(f"使用: --phe_col_num {phe_index}")
```

### 命令行检查
```bash
# 检查文件结构
head -3 your_file.txt
awk '{print NF; exit}' your_file.txt
```

## ⚠️ 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `IndexError: out-of-bounds` | 列索引错误 | 检查 `--phe_col_num` |
| `Invalid classes` | 分类标签不从0开始 | 转换标签为0,1,2,3... |
| `未知文件格式` | 扩展名不支持 | 使用.raw/.txt/.ped/.vcf |

## 📋 支持的文件格式

| 格式 | 扩展名 | 基因型列 | 表型列 |
|------|--------|----------|--------|
| PLINK RAW | .raw | 第7列开始 | 第6列 |
| 文本格式 | .txt | 第2列开始 | 需指定 |
| PLINK PED | .ped | 第7列开始 | 第6列 |
| VCF格式 | .vcf | 第10列开始 | 需指定 |

## 🎯 支持的模型

### 回归任务
- `xgboost` - XGBoost回归
- `LGBMRegressor` - LightGBM回归
- `CatBoostRegressor` - CatBoost回归
- `RandomForest` - 随机森林回归
- `SVM` - 支持向量回归

### 分类任务
- `xgboost` - XGBoost分类
- `LGBM` - LightGBM分类
- `CatBoost` - CatBoost分类
- `RandomForest` - 随机森林分类
- `LogisticRegression` - 逻辑回归

## 📊 评估指标

| 任务类型 | 主要指标 | 取值范围 |
|----------|----------|----------|
| 回归 | Pearson相关系数 | -1 到 1 |
| 分类 | 准确率 | 0 到 1 |

---

💡 **详细说明请查看**: `AIGP_数据格式和索引说明.md`
