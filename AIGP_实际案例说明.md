# AIGP 实际案例说明

## 🐎 案例：马匹基因组表型预测

### 数据文件说明

#### 1. 基因型文件：`horse_480_use.raw`
```
FID IID PAT MAT SEX PHENOTYPE SNP1 SNP2 SNP3 SNP4 ...
1   1   0   0   1   2         1    0    2    1    ...
2   2   0   0   2   1         0    1    1    2    ...
3   3   0   0   1   2         2    0    0    1    ...
```

**文件结构分析**：
- **总列数**：482列（1个样本ID + 481个SNP特征）
- **基因型数据**：从第7列开始（索引5开始）
- **特征数量**：476个SNP特征

#### 2. 表型文件：`horse_phe_0indexed.txt`
```
FID IID PHENOTYPE
1   1   0
2   2   2
3   3   0
```

**文件结构分析**：
- **总列数**：3列
- **表型数据**：第3列（PHENOTYPE）
- **分类标签**：0, 1, 2, 3（4分类任务）

### 索引计算过程

#### 步骤1：确定原始列位置
- PHENOTYPE在原始文件的第3列
- 原始列位置 = 3

#### 步骤2：直接使用原始位置
```
实际索引 = 原始列位置
实际索引 = 3
```

#### 步骤3：验证数据加载
```python
import pandas as pd

# 读取表型文件
df = pd.read_csv('horse_phe_0indexed.txt', sep=' ', header=0)
print("原始列:", list(df.columns))
# 输出: ['FID', 'IID', 'PHENOTYPE']

# 设置FID为索引
df_indexed = df.set_index('FID')
print("设置索引后:", list(df_indexed.columns))
# 输出: ['IID', 'PHENOTYPE']

# 检查PHENOTYPE列索引
phe_index = df_indexed.columns.get_loc('PHENOTYPE')
print(f"PHENOTYPE列索引: {phe_index}")
# 输出: PHENOTYPE列索引: 1
```

**重要发现**：实际索引是3，直接使用原始列位置！

#### 步骤4：确认索引
- 原始位置：第3列
- 实际索引：3 ✅
- 原因：第一列是默认索引，后续列位置就是实际索引

### 正确的命令

#### 1. 基本分类任务
```bash
python main.py \
  --geno "horse_480_use.raw" \
  --phe "horse_phe_0indexed.txt" \
  --phe_sep " " \
  --phe_col_num 3 \
  --type classification \
  --model xgboost
```

#### 2. 10折交叉验证
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

#### 3. PCA降维（100维）
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

#### 4. PHATE降维（100维）
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

### 结果分析

#### PCA + XGBoost 结果
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

#### PHATE + XGBoost 结果
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

### 性能对比

| 方法 | 平均准确率 | 标准差 | 性能评价 |
|------|------------|--------|----------|
| PCA + XGBoost | 66.25% | 2.60% | 较好 |
| PHATE + XGBoost | 62.08% | 4.15% | 一般 |

**分析**：
- PCA降维效果更好
- 随机猜测准确率：25%（4分类）
- 两个方法都显著优于随机水平

## 🔍 调试技巧

### 1. 自动索引检查脚本
```python
def check_phenotype_index(file_path, sep=' '):
    """自动检查表型文件索引"""
    df = pd.read_csv(file_path, sep=sep, header=0)
    df_indexed = df.set_index(df.columns[0])
    
    print(f"文件: {file_path}")
    print(f"原始列: {list(df.columns)}")
    print(f"设置索引后: {list(df_indexed.columns)}")
    
    if 'PHENOTYPE' in df_indexed.columns:
        index = df_indexed.columns.get_loc('PHENOTYPE')
        print(f"PHENOTYPE列索引: {index}")
        print(f"使用命令: --phe_col_num {index}")
    else:
        print("未找到PHENOTYPE列")
        print("可用列:", list(df_indexed.columns))

# 使用示例
check_phenotype_index('horse_phe_0indexed.txt', ' ')
```

### 2. 数据质量检查
```python
def check_data_quality(geno_file, phe_file):
    """检查数据质量"""
    # 检查基因型数据
    geno_df = pd.read_csv(geno_file, sep=r'\s+', header=0)
    geno_df = geno_df.set_index('IID')
    X = geno_df.iloc[:, 5:]
    
    print("基因型数据:")
    print(f"  样本数: {len(X)}")
    print(f"  特征数: {len(X.columns)}")
    print(f"  缺失值: {X.isnull().sum().sum()}")
    
    # 检查表型数据
    phe_df = pd.read_csv(phe_file, sep=' ', header=0)
    phe_df = phe_df.set_index('FID')
    y = phe_df.iloc[:, 1]
    
    print("表型数据:")
    print(f"  样本数: {len(y)}")
    print(f"  类别分布: {y.value_counts().sort_index()}")
    print(f"  缺失值: {y.isnull().sum()}")

# 使用示例
check_data_quality('horse_480_use.raw', 'horse_phe_0indexed.txt')
```

## 📋 常见问题解决

### 问题1：索引错误
**错误信息**：`IndexError: single positional indexer is out-of-bounds`

**解决步骤**：
1. 使用调试脚本检查实际索引
2. 确认转换公式
3. 验证数据加载过程

### 问题2：分类标签错误
**错误信息**：`ValueError: Invalid classes inferred from unique values`

**解决方案**：
```python
# 转换分类标签
df['PHENOTYPE'] = df['PHENOTYPE'] - 1  # 1,2,3,4 → 0,1,2,3
```

### 问题3：文件格式错误
**错误信息**：`ValueError: 未知的基因型文件格式`

**解决方案**：
- 检查文件扩展名
- 使用支持的格式：.raw, .txt, .ped, .vcf

## 🎯 最佳实践总结

### 1. 数据准备
- 使用PLINK RAW格式
- 确保样本ID一致
- 检查数据质量

### 2. 索引计算
- 使用调试脚本验证
- 不要盲目应用公式
- 检查实际加载结果

### 3. 命令测试
- 先用小数据集测试
- 检查输出结果
- 验证交叉验证结果

### 4. 结果分析
- 比较不同方法
- 检查性能稳定性
- 验证生物学意义

---

💡 **提示**：遇到问题时，请使用提供的调试脚本检查您的数据格式和索引设置。
