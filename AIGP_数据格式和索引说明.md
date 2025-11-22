# AIGP 数据格式和索引详细说明

## 📖 项目概述

AIGP (Automated Genomic Phenotype Prediction) 是一个基于机器学习的基因组表型预测工具，支持多种数据格式和机器学习算法。

## 🧬 基因型数据格式和索引

### 1. PLINK RAW格式 (.raw) - 推荐格式

#### 文件结构
```
FID IID PAT MAT SEX PHENOTYPE SNP1_1 SNP2_1 SNP3_2 SNP4_0 ...
1   1   0   0   1   2         1      0      2      1     ...
2   2   0   0   2   1         0      1      1      2     ...
3   3   0   0   1   2         2      0      0      1     ...
```

#### 列索引说明
| 列位置 | 列名 | 内容 | 索引 | 说明 |
|--------|------|------|------|------|
| 0 | FID | 家系ID | - | 设为索引 |
| 1 | IID | 个体ID | - | 设为索引 |
| 2 | PAT | 父本ID | - | 忽略 |
| 3 | MAT | 母本ID | - | 忽略 |
| 4 | SEX | 性别 | - | 忽略 |
| 5 | PHENOTYPE | 表型 | - | 忽略（使用单独表型文件） |
| 6+ | SNP1, SNP2... | 基因型数据 | 0, 1, 2... | **实际使用的特征** |

#### 数据加载过程
```python
# 1. 读取文件
df = pd.read_csv('genotype.raw', sep=r'\s+', header=0)

# 2. 设置IID为索引
df = df.set_index('IID')

# 3. 提取基因型数据（从第6列开始，索引5开始）
X = df.iloc[:, 5:]  # 所有SNP特征
```

**结果**：基因型数据从索引0开始，包含所有SNP特征

### 2. 文本格式 (.txt)

#### 文件结构
```
Sample_ID SNP1 SNP2 SNP3 SNP4 ...
sample1   0    1    2    1    ...
sample2   1    0    1    2    ...
sample3   2    0    0    1    ...
```

#### 列索引说明
| 列位置 | 列名 | 内容 | 索引 | 说明 |
|--------|------|------|------|------|
| 0 | Sample_ID | 样本ID | - | 设为索引 |
| 1+ | SNP1, SNP2... | 基因型数据 | 0, 1, 2... | **实际使用的特征** |

#### 数据加载过程
```python
# 1. 读取文件
df = pd.read_csv('genotype.txt', sep=',', header=0)

# 2. 设置第一列为索引
df = df.set_index(df.columns[0])

# 3. 提取所有列作为特征
X = df  # 所有列都是特征
```

**结果**：所有列都是基因型特征，从索引0开始

## 🎯 表型数据格式和索引

### 1. 标准表型文件格式

#### 文件结构
```
FID IID PHENOTYPE
1   1   3.45
2   2   2.18
3   3   4.92
```

#### 列索引说明
| 列位置 | 列名 | 内容 | 索引 | 说明 |
|--------|------|------|------|------|
| 0 | FID | 家系ID | - | 设为索引 |
| 1 | IID | 个体ID | - | 忽略 |
| 2 | PHENOTYPE | 表型值 | 0 | **目标变量** |

#### 数据加载过程
```python
# 1. 读取文件
df = pd.read_csv('phenotype.txt', sep=' ', header=0)

# 2. 设置FID为索引
df = df.set_index('FID')

# 3. 提取表型数据（第3列，索引1）
y = df.iloc[:, 1]  # PHENOTYPE列
```

**结果**：表型数据在索引1，使用 `--phe_col_num 1`

### 2. 多列表型文件格式

#### 文件结构
```
FID IID AGE SEX PHENOTYPE WEIGHT HEIGHT
1   1   25  M   3.45      70     175
2   2   30  F   2.18      65     160
3   3   28  M   4.92      80     180
```

#### 列索引说明
| 列位置 | 列名 | 内容 | 索引 | 说明 |
|--------|------|------|------|------|
| 0 | FID | 家系ID | - | 设为索引 |
| 1 | IID | 个体ID | - | 忽略 |
| 2 | AGE | 年龄 | 0 | 协变量 |
| 3 | SEX | 性别 | 1 | 协变量 |
| 4 | PHENOTYPE | 表型值 | 2 | **目标变量** |
| 5 | WEIGHT | 体重 | 3 | 协变量 |
| 6 | HEIGHT | 身高 | 4 | 协变量 |

#### 数据加载过程
```python
# 1. 读取文件
df = pd.read_csv('phenotype.txt', sep=' ', header=0)

# 2. 设置FID为索引
df = df.set_index('FID')

# 3. 提取表型数据（第5列，索引2）
y = df.iloc[:, 2]  # PHENOTYPE列
```

**结果**：表型数据在索引2，使用 `--phe_col_num 2`

## 🔢 索引计算规则详解

### 核心规则
```
实际索引 = 原始列位置
```

**重要说明**：第一列是默认索引，后续列的位置就是实际索引，不需要减1！

### 详细说明

#### 步骤1：确定原始列位置
- 从1开始计数（人类习惯）
- 例如：PHENOTYPE在第3列，原始位置 = 3

#### 步骤2：直接使用原始位置
- 实际索引 = 3
- 使用 `--phe_col_num 3`

#### 步骤3：验证结果
- 检查加载后的DataFrame结构
- 确认目标列在正确位置

## 📊 实际案例演示

### 案例1：horse_phe_0indexed.txt

#### 文件内容
```
FID IID PHENOTYPE
1   1   0
2   2   2
3   3   0
```

#### 索引计算
1. **原始列位置**：PHENOTYPE在第3列
2. **直接使用**：第3列就是索引3
3. **实际索引**：3
4. **命令参数**：`--phe_col_num 3`

#### 数据加载验证
```python
import pandas as pd

# 读取文件
df = pd.read_csv('horse_phe_0indexed.txt', sep=' ', header=0)
print("原始列:", list(df.columns))
# 输出: ['FID', 'IID', 'PHENOTYPE']

# 设置索引
df_indexed = df.set_index('FID')
print("设置索引后:", list(df_indexed.columns))
# 输出: ['IID', 'PHENOTYPE']

# 检查PHENOTYPE列索引
phe_index = df_indexed.columns.get_loc('PHENOTYPE')
print(f"PHENOTYPE列索引: {phe_index}")
# 输出: PHENOTYPE列索引: 1
```

**结果**：实际应该使用 `--phe_col_num 3`（第3列）

### 案例2：多列表型文件

#### 文件内容
```
FID IID COV1 COV2 PHENOTYPE OTHER_TRAITS
1   1   25   M    3.45      trait1
2   2   30   F    2.18      trait2
```

#### 索引计算
1. **原始列位置**：PHENOTYPE在第5列
2. **直接使用**：第5列就是索引5
3. **实际索引**：5
4. **命令参数**：`--phe_col_num 5`

### 案例3：基因型文件

#### PLINK RAW格式
```
FID IID PAT MAT SEX PHENOTYPE SNP1 SNP2 SNP3
1   1   0   0   1   2         1    0    2
2   2   0   0   2   1         0    1    1
```

#### 基因型数据提取
- **基因型数据**：从第7列开始（SNP1, SNP2, SNP3...）
- **索引范围**：0, 1, 2, 3...
- **自动处理**：系统自动提取所有SNP列

## 🛠️ 调试工具和验证方法

### Python调试脚本

```python
def debug_data_format(geno_file, phe_file, geno_sep=',', phe_sep=' '):
    """
    调试数据格式和索引
    """
    print("=" * 60)
    print("AIGP 数据格式调试工具")
    print("=" * 60)
    
    # 1. 检查基因型文件
    print("\n1. 基因型文件分析:")
    print("-" * 30)
    try:
        geno_df = pd.read_csv(geno_file, sep=geno_sep, header=0)
        print(f"文件: {geno_file}")
        print(f"列名: {list(geno_df.columns)}")
        print(f"列数: {len(geno_df.columns)}")
        print(f"样本数: {len(geno_df)}")
        
        # 设置索引
        geno_indexed = geno_df.set_index(geno_df.columns[0])
        print(f"设置索引后列数: {len(geno_indexed.columns)}")
        print(f"基因型特征数: {len(geno_indexed.columns)}")
        
    except Exception as e:
        print(f"错误: {e}")
    
    # 2. 检查表型文件
    print("\n2. 表型文件分析:")
    print("-" * 30)
    try:
        phe_df = pd.read_csv(phe_file, sep=phe_sep, header=0)
        print(f"文件: {phe_file}")
        print(f"列名: {list(phe_df.columns)}")
        print(f"列数: {len(phe_df.columns)}")
        print(f"样本数: {len(phe_df)}")
        
        # 设置索引
        phe_indexed = phe_df.set_index(phe_df.columns[0])
        print(f"设置索引后列名: {list(phe_indexed.columns)}")
        
        # 查找PHENOTYPE列
        if 'PHENOTYPE' in phe_indexed.columns:
            phe_index = phe_indexed.columns.get_loc('PHENOTYPE')
            print(f"PHENOTYPE列索引: {phe_index}")
            print(f"建议使用: --phe_col_num {phe_index}")
        else:
            print("未找到PHENOTYPE列")
            print("可用列:", list(phe_indexed.columns))
            
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n" + "=" * 60)

# 使用示例
debug_data_format('horse_480_use.raw', 'horse_phe_0indexed.txt', ' ', ' ')
```

### 命令行快速检查

```bash
# 检查基因型文件
head -3 horse_480_use.raw
awk '{print NF; exit}' horse_480_use.raw

# 检查表型文件
head -3 horse_phe_0indexed.txt
awk '{print NF; exit}' horse_phe_0indexed.txt
```

## 🎯 常用命令示例

### 1. 基本回归任务
```bash
python main.py \
  --geno "horse_480_use.raw" \
  --phe "horse_phe_0indexed.txt" \
  --phe_sep " " \
  --phe_col_num 1 \
  --type regression \
  --model xgboost
```

### 2. 分类任务（10折交叉验证）
```bash
python main.py \
  --geno "horse_480_use.raw" \
  --phe "horse_phe_0indexed.txt" \
  --phe_sep " " \
  --phe_col_num 1 \
  --type classification \
  --model xgboost \
  --cv 10 \
  --result_file "results.txt"
```

### 3. 使用PCA降维
```bash
python main.py \
  --geno "horse_480_use.raw" \
  --phe "horse_phe_0indexed.txt" \
  --phe_sep " " \
  --phe_col_num 1 \
  --type classification \
  --model xgboost \
  --dim_reduction pca \
  --n_components 100 \
  --cv 10
```

### 4. 使用PHATE降维
```bash
python main.py \
  --geno "horse_480_use.raw" \
  --phe "horse_phe_0indexed.txt" \
  --phe_sep " " \
  --phe_col_num 1 \
  --type classification \
  --model xgboost \
  --dim_reduction phate \
  --n_components 100 \
  --cv 10
```

## ⚠️ 常见错误和解决方案

### 错误1：IndexError: single positional indexer is out-of-bounds

**原因**：`--phe_col_num` 参数超出范围

**解决方案**：
1. 使用调试脚本检查实际列数
2. 确认转换公式：`实际索引 = 原始列位置 - 1`
3. 检查文件格式和分隔符

### 错误2：ValueError: Invalid classes inferred from unique values

**原因**：分类标签不从0开始

**解决方案**：
```python
# 转换分类标签
df['PHENOTYPE'] = df['PHENOTYPE'] - 1  # 1,2,3,4 → 0,1,2,3
```

### 错误3：ValueError: 未知的基因型文件格式

**原因**：文件扩展名不支持

**解决方案**：
- 使用支持的格式：.raw, .txt, .ped, .vcf
- 检查文件扩展名

## 📋 快速参考表

### 表型列索引快速查找

| 原始列位置 | 实际索引 | 命令参数 | 示例 |
|------------|----------|----------|------|
| 第2列 | 0 | `--phe_col_num 0` | IID列 |
| 第3列 | 1 | `--phe_col_num 1` | PHENOTYPE列 |
| 第4列 | 2 | `--phe_col_num 2` | 协变量列 |
| 第5列 | 3 | `--phe_col_num 3` | 协变量列 |
| 第6列 | 4 | `--phe_col_num 4` | 协变量列 |

### 文件格式支持

| 格式 | 扩展名 | 基因型列 | 表型列 | 说明 |
|------|--------|----------|--------|------|
| PLINK RAW | .raw | 第7列开始 | 第6列 | 推荐格式 |
| 文本格式 | .txt | 第2列开始 | 需指定 | 灵活格式 |
| PLINK PED | .ped | 第7列开始 | 第6列 | 需转换 |
| VCF格式 | .vcf | 第10列开始 | 需指定 | 需转换 |

## 🔧 最佳实践

### 1. 数据准备
- 使用PLINK RAW格式（推荐）
- 确保样本ID一致
- 检查数据质量

### 2. 索引计算
- 使用调试脚本验证
- 记住转换公式
- 检查加载后的结构

### 3. 命令测试
- 先用小数据集测试
- 检查输出结果
- 验证交叉验证结果

---

💡 **提示**：遇到问题时，请使用提供的调试脚本检查您的数据格式和索引设置。
