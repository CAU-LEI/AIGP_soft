# AIGP 说明文档总览

## 📚 文档结构

本说明文档包含以下4个部分，帮助您快速掌握AIGP工具的使用：

### 1. 📖 [AIGP_数据格式和索引说明.md](AIGP_数据格式和索引说明.md)
**详细的技术说明文档**
- 基因型和表型数据格式详解
- 索引计算规则和转换公式
- 数据加载过程说明
- 调试工具和验证方法
- 常见错误和解决方案

### 2. 🚀 [AIGP_快速参考卡.md](AIGP_快速参考卡.md)
**快速查阅的参考卡片**
- 基本命令格式
- 常用参数说明
- 快速命令示例
- 常见错误速查表

### 3. 🐎 [AIGP_实际案例说明.md](AIGP_实际案例说明.md)
**基于真实数据的案例演示**
- 马匹基因组表型预测案例
- 详细的索引计算过程
- 完整的命令示例
- 结果分析和性能对比

### 4. 📋 [README_AIGP_说明文档.md](README_AIGP_说明文档.md)
**本文档 - 总览和导航**

## 🎯 快速开始

### 如果您是新手用户
1. 先阅读 [AIGP_数据格式和索引说明.md](AIGP_数据格式和索引说明.md) 了解基本概念
2. 参考 [AIGP_实际案例说明.md](AIGP_实际案例说明.md) 学习具体操作
3. 使用 [AIGP_快速参考卡.md](AIGP_快速参考卡.md) 进行日常查询

### 如果您是有经验的用户
1. 直接查看 [AIGP_快速参考卡.md](AIGP_快速参考卡.md) 获取命令格式
2. 参考 [AIGP_实际案例说明.md](AIGP_实际案例说明.md) 了解最佳实践

## 🔑 核心要点

### 数据格式要求
- **基因型数据**：支持PLINK RAW、文本、PED、VCF格式
- **表型数据**：标准格式，第一列为样本ID
- **分类任务**：标签必须从0开始（0,1,2,3...）

### 索引计算规则
```
实际索引 = 原始列位置 - 1
```

**重要**：数据加载后第一列会被设为索引，后续列索引会相应调整

### 常用命令格式
```bash
python main.py \
  --geno <基因型文件> \
  --phe <表型文件> \
  --phe_sep " " \
  --phe_col_num <表型列索引> \
  --type <任务类型> \
  --model <模型名称> \
  --cv <交叉验证折数>
```

## 📊 支持的功能

### 机器学习算法
- **回归任务**：XGBoost, LightGBM, CatBoost, RandomForest, SVM等
- **分类任务**：XGBoost, LightGBM, CatBoost, RandomForest, LogisticRegression等

### 降维方法
- **PCA**：主成分分析
- **PHATE**：基于热扩散的降维

### 优化方法
- **网格搜索**：Grid Search
- **麻雀搜索算法**：Sparrow Search Algorithm

### 分析功能
- **交叉验证**：K折交叉验证
- **SHAP分析**：模型解释性分析
- **特征重要性**：特征选择和分析

## 🛠️ 调试工具

### Python调试脚本
```python
import pandas as pd

def check_phenotype_index(file_path, sep=' '):
    """自动检查表型文件索引"""
    df = pd.read_csv(file_path, sep=sep, header=0)
    df_indexed = df.set_index(df.columns[0])
    
    if 'PHENOTYPE' in df_indexed.columns:
        index = df_indexed.columns.get_loc('PHENOTYPE')
        print(f"使用命令: --phe_col_num {index}")
    else:
        print("未找到PHENOTYPE列")

# 使用示例
check_phenotype_index('your_phenotype_file.txt', ' ')
```

### 命令行检查
```bash
# 检查文件结构
head -3 your_file.txt
awk '{print NF; exit}' your_file.txt
```

## ⚠️ 常见问题

### 1. 索引错误
**错误**：`IndexError: single positional indexer is out-of-bounds`
**解决**：使用调试脚本检查实际索引

### 2. 分类标签错误
**错误**：`ValueError: Invalid classes inferred from unique values`
**解决**：确保分类标签从0开始

### 3. 文件格式错误
**错误**：`ValueError: 未知的基因型文件格式`
**解决**：使用支持的格式（.raw, .txt, .ped, .vcf）

## 📈 性能基准

### 马匹基因组预测案例
| 方法 | 平均准确率 | 标准差 | 性能评价 |
|------|------------|--------|----------|
| PCA + XGBoost | 66.25% | 2.60% | 较好 |
| PHATE + XGBoost | 62.08% | 4.15% | 一般 |

**说明**：随机猜测准确率为25%（4分类任务），两个方法都显著优于随机水平。

## 🔄 更新日志

### v2.0.0 (当前版本)
- ✅ 重新设计说明文档结构
- ✅ 添加详细的索引计算说明
- ✅ 提供实际案例演示
- ✅ 完善调试工具和验证方法
- ✅ 优化快速参考卡

### v1.0.0
- ✅ 初始版本发布
- ✅ 支持多种机器学习算法
- ✅ 集成降维和优化功能

## 📞 技术支持

如果您在使用过程中遇到问题：

1. **查阅文档**：先查看相关说明文档
2. **使用调试工具**：运行提供的调试脚本
3. **检查常见问题**：参考常见问题解决方案
4. **提交Issue**：在项目仓库提交问题报告

## 📄 许可证

本工具仅供学术研究使用，请确保遵守相关数据使用协议和法律法规。

---

💡 **提示**：建议收藏 [AIGP_快速参考卡.md](AIGP_快速参考卡.md) 以便快速查阅常用命令和参数。
