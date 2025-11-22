# AIGP 数据格式规范

## 概述

本文档详细说明了 AIGP (Automated Genomic Phenotype Prediction) 框架支持的基因型和表型数据格式。

## 基因型数据格式

### 支持的文件格式

1. **PLINK .raw 格式** (推荐)
2. **文本文件 (.txt)**
3. **PLINK .ped 格式**
4. **VCF 格式**

### PLINK .raw 格式 (标准格式)

#### 文件结构
```
FID    IID    PAT    MAT    SEX    PHENOTYPE    SNP1_1    SNP2_1    SNP3_2    ...
FAM001 IND001 0      0      1      2           0         1         2         ...
FAM002 IND002 0      0      2      1           1         0         1         ...
```

#### 列说明
- **FID**: 家系ID (Family ID)
- **IID**: 个体ID (Individual ID) - 用作样本索引
- **PAT**: 父亲ID (Paternal ID)
- **MAT**: 母亲ID (Maternal ID)  
- **SEX**: 性别 (1=男性, 2=女性, 其他=未知)
- **PHENOTYPE**: 表型值
- **SNP列**: 基因型数据，每个SNP一列，值为 0/1/2 (加性编码)

#### 索引规则
- 第一列 (FID) 被设置为pandas DataFrame的索引
- 基因型数据从第7列开始 (索引6，0-based)
- 表型数据在第6列 (索引5，0-based)，但设置FID为索引后变为索引4

### 文本文件格式

#### 基本要求
- 必须有表头行
- 第一列作为样本ID
- 支持空格、制表符、逗号等分隔符

#### 示例
```
SampleID    Feature1    Feature2    Feature3    Phenotype
SAMPLE001   0.1         0.2         0.3         1
SAMPLE002   0.4         0.5         0.6         0
```

## 表型数据格式

### 独立表型文件

当基因型和表型数据分开存储时：

```
SampleID    Trait1    Trait2    Target
SAMPLE001   1.2       3.4       1
SAMPLE002   2.1       4.3       0
```

### 嵌入式表型 (PLINK .raw)

表型数据包含在基因型文件的PHENOTYPE列中。

## 数据类型和任务类型

### 回归任务 (regression)
- 表型值：连续数值
- 示例：身高、体重、产量等

### 分类任务 (classification)  
- 表型值：离散类别
- 二分类：0/1 或 1/2
- 多分类：0/1/2/... 或 1/2/3/...

## 命令行参数说明

### 基本参数
- `--geno`: 基因型文件路径
- `--geno_sep`: 基因型文件分隔符 (默认: ",")
- `--phe`: 表型文件路径 (可选，如果基因型文件包含表型)
- `--phe_sep`: 表型文件分隔符 (默认: ",")
- `--phe_col_num`: 表型列索引 (0-based)

### 索引计算规则

#### PLINK .raw 格式
```python
# 原始列结构: FID, IID, PAT, MAT, SEX, PHENOTYPE, SNP1, SNP2, ...
# 列索引:      0    1    2    3    4    5          6     7     ...

# 设置FID为索引后: IID, PAT, MAT, SEX, PHENOTYPE, SNP1, SNP2, ...
# 新索引:            0    1    2    3    4          5     6     ...

# 因此:
# - 表型列从索引5变为索引4
# - 基因型数据从索引6变为索引5开始
```

#### 指定参数
```bash
# 对于 PLINK .raw 格式
python main.py --geno data.raw --geno_sep " " --phe data.raw --phe_sep " " --phe_col_num 4

# 对于独立表型文件
python main.py --geno geno.txt --phe pheno.txt --phe_col_num 2
```

## 数据预处理

### 样本对齐
- 基因型和表型数据通过样本ID自动对齐
- 只保留两个文件中都存在的样本 (inner join)

### 数据清洗
- **回归任务**: 移除表型值为空、空字符串或0的样本
- **分类任务**: 移除表型值为空或空字符串的样本

## 输出格式

### 交叉验证结果
使用 `--result_file` 参数可将交叉验证结果保存到指定文件：

```
Cross validation time: 3.43 sec

=== 10-Fold Cross Validation Results ===
Fold  1: 0.5417
Fold  2: 0.5208
...
Fold 10: 0.6458
----------------------------------------
Mean:     0.5542
Std Dev:  0.0605
Min:      0.4375
Max:      0.6458
========================================
```

## 常见问题

### Q: 为什么表型列索引需要-1？
A: 当使用相同文件作为基因型和表型数据源时，pandas会将第一列设置为索引，导致后续列的索引向前移动1位。

### Q: 支持哪些分隔符？
A: 支持空格(" ")、制表符("\t")、逗号(",")等常见分隔符。对于.raw文件，建议使用空格。

### Q: 如何处理缺失值？
A: 基因型数据中的缺失值通常编码为-9或NA，框架会自动处理。表型数据的缺失值会在数据清洗阶段移除对应样本。

## 示例命令

```bash
# 基本分类任务
python main.py --geno horse_480_use.raw --geno_sep " " --phe horse_480_use.raw --phe_sep " " --phe_col_num 4 --type classification --model xgboost --cv 10

# 带PCA降维的回归任务
python main.py --geno data.txt --phe pheno.txt --phe_col_num 1 --type regression --model LinearRegression --dim_reduction pca --n_components 50 --cv 5

# 结果输出到文件
python main.py --geno data.raw --geno_sep " " --phe data.raw --phe_sep " " --phe_col_num 4 --type classification --model xgboost --cv 10 --result_file results.txt
```
