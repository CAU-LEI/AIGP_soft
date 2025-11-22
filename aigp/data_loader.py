# aigp/data_loader.py
import os
import subprocess
import pandas as pd

def load_genotype_data(geno_path, geno_sep):
    """
    根据基因型文件后缀自动检测格式并读取：
    """
    ext = os.path.splitext(geno_path)[1].lower()
    if ext == ".txt":
        df = pd.read_csv(geno_path, sep=geno_sep, header=0)
    elif ext == ".ped":
        base = geno_path[:-4]
        cmd = ["plink", "--file", base, "--recode", "A", "--out", base + "_recode"]
        subprocess.run(cmd, check=True)
        recoded_file = base + "_recode.raw"
        df = pd.read_csv(recoded_file, delim_whitespace=True, header=0)
    elif ext == ".vcf":
        base = geno_path[:-4]
        cmd = ["plink", "--vcf", geno_path, "--recode", "ped", "--out", base + "_vcf"]
        subprocess.run(cmd, check=True)
        ped_file = base + "_vcf.ped"
        base2 = ped_file[:-4]
        cmd = ["plink", "--file", base2, "--recode", "A", "--out", base2 + "_recode"]
        subprocess.run(cmd, check=True)
        recoded_file = base2 + "_recode.raw"
        df = pd.read_csv(recoded_file, delim_whitespace=True, header=0)
    elif ext == ".raw":
        df = pd.read_csv(geno_path, sep=r"\s+", header=0)
    else:
        raise ValueError("未知的基因型文件格式: {}".format(ext))

    # ✅ 设置样本号为索引
    if "IID" in df.columns:
        # PLINK格式：去除前6列
        df = df.set_index("IID")
        X = df.iloc[:, 5:]  # 去除 FID、PAT、MAT、SEX、PHENOTYPE，共6列
    else:
        # 简单格式：第一列作为样本ID
        df = df.set_index(df.columns[0])
        X = df  # 使用所有列作为特征

    return X


def auto_detect_phe_col(phe_df):
    """
    自动检测表型标签列：
      - 优先查找标题中包含 "phenotype" 或 "trait" 的列（不区分大小写）；
      - 若无则返回第一个数值型的列；
      - 否则返回第 0 列。
    """
    for i, col in enumerate(phe_df.columns):
        if "phenotype" in col.lower() or "trait" in col.lower():
            return i
    for i, col in enumerate(phe_df.columns):
        try:
            pd.to_numeric(phe_df[col])
            return i
        except:
            continue
    return 0


def load_training_data(geno_path, geno_sep, phe_path, phe_sep, phe_col_num, category_cols=None, task_type="regression"):
    """
    读取训练数据：保留样本号索引，自动对齐样本。
    """
    X = load_genotype_data(geno_path, geno_sep)

    if phe_path is not None:
        phe_df = pd.read_csv(phe_path, sep=phe_sep, header=0)
        phe_df = phe_df.set_index(phe_df.columns[0])  # ✅ 用第一列设为样本号
    else:
        phe_df = None

    if phe_df is not None and phe_col_num is None:
        phe_col_num = auto_detect_phe_col(phe_df)

    if phe_df is not None and phe_col_num is not None:
        # 最常规的习惯：用户输入第几列，就直接用第几列
        # 但设置索引后列数会减少1，所以需要减去1
        y = phe_df.iloc[:, phe_col_num - 2]  # 转换为0-based索引（减去2：1个索引列+1个0-based转换）
    else:
        y = None

    # ✅ 对齐索引（样本号），保持交集样本
    if y is not None:
        X, y = X.align(y, join="inner", axis=0)

        if task_type == "regression":
            valid_idx = y.notna() & (y != "") & (y != 0)
        else:
            valid_idx = y.notna() & (y != "")
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

    # 协变量
    covariates = None
    if phe_df is not None and category_cols is not None:
        covariates = phe_df.iloc[:, category_cols]
        covariates = covariates.loc[X.index]  # ✅ 也对齐样本顺序

    return X, y, covariates


def load_candidate_data(geno_path, geno_sep, phe_path=None, phe_sep=None, category_cols=None):
    """
    读取候选群体数据
    """
    X = load_genotype_data(geno_path, geno_sep)
    covariates = None
    if phe_path is not None and category_cols is not None:
        phe_df = pd.read_csv(phe_path, sep=phe_sep, header=0)
        covariates = phe_df.iloc[:, category_cols]
    return X, covariates


def calculate_geno_stats(X):
    """
    计算基因型数据的统计信息：对每个标记计算缺失率、等位基因频率和最小等位基因频率（MAF）
    返回一个 DataFrame，并可保存为 CSV。
    """
    stats = []
    for col in X.columns:
        series = X[col]
        missing_rate = series.isna().mean()
        nonmissing = series.dropna()
        if len(nonmissing) == 0:
            freq = None
            maf = None
        else:
            # 假定基因型编码为 0,1,2
            freq = nonmissing.sum() / (2 * len(nonmissing))
            maf = min(freq, 1 - freq)
        stats.append({"marker": col, "missing_rate": missing_rate, "allele_frequency": freq, "MAF": maf})
    return pd.DataFrame(stats)


def calculate_phe_stats(y):
    """
    计算表型数据的均值和标准差，并画出分布图保存到当前目录
    """
    import matplotlib.pyplot as plt
    if y.dtype.kind in 'biufc':  # 数值型数据
        mean_val = y.mean()
        std_val = y.std()
        plt.figure()
        y.hist(bins=30)
        plt.title("Phenotype Distribution")
        plt.xlabel("Phenotype")
        plt.ylabel("Frequency")
        plt.savefig("phe_distribution.png")
        plt.close()
        print("表型数据均值: {:.4f}, 标准差: {:.4f}".format(mean_val, std_val))
        return mean_val, std_val
    else:
        print("表型数据不是数值型，无法计算均值和标准差。")
        return None, None
