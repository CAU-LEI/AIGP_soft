#!/usr/bin/env python
# predict.py
import os
import pandas as pd
import joblib
import argparse
from aigp.data_loader import load_candidate_data
from aigp.dim_reduction import reduce_dimensions


def parse_args():
    parser = argparse.ArgumentParser(description="Candidate Population Prediction")
    parser.add_argument("--geno", type=str, required=True,
                        help="Candidate genotype file path")
    parser.add_argument("--geno_sep", type=str, default=",",
                        help="Candidate genotype file separator")
    parser.add_argument("--phe", type=str, default=None,
                        help="Candidate phenotype/covariate file path (if needed)")
    parser.add_argument("--phe_sep", type=str, default=",",
                        help="Candidate phenotype file separator")
    parser.add_argument("--category_cols", type=str, default=None,
                        help="Covariate column indices (e.g., 1,2)")
    parser.add_argument("--dim_reduction", type=str, choices=["pca", "phate"], default=None,
                        help="Dimension reduction method used in training")
    parser.add_argument("--n_components", type=int, default=None,
                        help="Number of components for dimension reduction")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model checkpoint")
    args = parser.parse_args()
    if args.category_cols:
        args.category_cols = [int(x.strip()) for x in args.category_cols.split(",")]
    return args


def main():
    args = parse_args()
    # 加载候选群体数据（如果有协变量文件则一并加载）
    X, covariates = load_candidate_data(args.geno, args.geno_sep, args.phe, args.phe_sep, args.category_cols)

    # 如果训练时使用了降维，则对候选数据也进行降维
    if args.dim_reduction and args.n_components:
        X_reduced = reduce_dimensions(X, args.dim_reduction, args.n_components)
        X_reduced = pd.DataFrame(X_reduced, columns=["PC{}".format(i + 1) for i in range(args.n_components)])
        if covariates is not None:
            X = pd.concat([X_reduced, covariates.reset_index(drop=True)], axis=1)
        else:
            X = X_reduced
    elif covariates is not None:
        X = pd.concat([X.reset_index(drop=True), covariates.reset_index(drop=True)], axis=1)

    # 加载保存的模型检查点
    if not os.path.exists(args.model_path):
        raise ValueError("Model checkpoint not found: {}".format(args.model_path))
    model = joblib.load(args.model_path)

    # 进行预测
    predictions = model.predict(X)

    # 保存预测结果
    output_file = "candidate_predictions.csv"
    pd.DataFrame({"prediction": predictions}).to_csv(output_file, index=False)
    print("Predictions saved to:", output_file)


if __name__ == "__main__":
    main()
