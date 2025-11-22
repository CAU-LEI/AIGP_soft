# aigp/cli.py
import argparse
import json


def parse_args():
    """
    解析命令行参数，并返回解析结果
    """
    parser = argparse.ArgumentParser(description="AIGP: 基于机器学习的基因组表型预测软件")

    # 数据相关参数
    parser.add_argument("--geno", type=str, required=True,
                        help="输入特征数据文件的路径")
    parser.add_argument("--geno_sep", type=str, default=",",
                        help="输入特征数据文件的分隔符，默认为','")
    parser.add_argument("--phe", type=str, default=None,
                        help="输入标签数据文件的路径")
    parser.add_argument("--phe_sep", type=str, default=",",
                        help="输入标签数据文件的分隔符，默认为','")
    parser.add_argument("--phe_col_num", type=int, default=None,
                        help="标签数据所在的列号（0-indexed），若不指定则自动检测")
    parser.add_argument("--category_cols", type=str, default=None,
                        help="指定哪些列为分类变量（协变量），多个用逗号分隔，如：1,2")

    # 任务类型与模型相关参数
    parser.add_argument("--type", type=str, choices=["classification", "regression"], required=True,
                        help="任务类型，classification（分类）或 regression（回归）")
    parser.add_argument("--model", type=str, default=None,
                        help="选择的模型名称")
    parser.add_argument("--model_params", type=str, default="{}",
                        help="模型参数，JSON 格式字符串")

    # 降维参数
    parser.add_argument("--dim_reduction", type=str, choices=["pca", "phate"], default=None,
                        help="降维方法，可选 pca 或 phate，不选则使用原始数据")
    parser.add_argument("--n_components", type=int, default=None,
                        help="降维后的维度数")

    # 数据预处理参数（暂留接口）
    parser.add_argument("--process_x", action="store_true",
                        help="是否对特征数据进行预处理（待定功能）")
    parser.add_argument("--process_y", action="store_true",
                        help="是否对标签数据进行预处理（待定功能）")

    # 数据划分、交叉验证
    parser.add_argument("--cv", type=int, default=None,
                        help="交叉验证折数")
    parser.add_argument("--train_size", type=float, default=None,
                        help="训练集占比，例如 0.8 表示 80% 用于训练")
    parser.add_argument("--ntest", type=int, default=None,
                        help="测试集大小，指定前 n 个样本作为训练群体")

    # 超参数搜索选项
    parser.add_argument("--grid", action="store_true",
                        help="是否使用网格搜索调参")
    parser.add_argument("--grid_model_params", type=str, default="{}",
                        help="网格搜索的参数，JSON 格式字符串")
    parser.add_argument("--ssa", action="store_true",
                        help="是否使用 SSA 麻雀搜索调参")
    parser.add_argument("--ssa_model_params", type=str, default="{}",
                        help="SSA 搜索的参数，JSON 格式字符串，例如："
                             "{\"use_custom_ssa\": true, \"param_bounds\": {\"learning_rate\": [0.01, 0.3], "
                             "\"num_leaves\": [10, 50], \"max_depth\": [3, 10]}, \"pop_size\": 20, \"max_iter\": 30}")

    # SHAP 分析参数
    parser.add_argument("--shap", action="store_true",
                        help="是否计算并可视化 SHAP 值")
    parser.add_argument("--shap_beeswarm", action="store_true",
                        help="是否生成 SHAP 蜂群图（需 --shap）")
    parser.add_argument("--shap_feature_heatmap", action="store_true",
                        help="是否生成 SHAP 热图（需 --shap）")
    parser.add_argument("--shap_feature_waterfall", action="store_true",
                        help="是否生成 SHAP 瀑布图（需 --shap）")
    parser.add_argument("--top_features", type=int, default=None,
                        help="在 SHAP 图中显示的前 N 个特征（需 --shap）")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="指定保存 SHAP 图像的路径")
    parser.add_argument("--result_file", type=str, default=None,
                        help="指定交叉验证结果输出文件路径")

    # 候选群体预测相关
    parser.add_argument("--model_path", type=str, default=None,
                        help="指定参考群训练好的模型参数文件，用于候选群体预测")
    # 新增：基因型和表型统计计算及特征重要性输出
    parser.add_argument("--geno_cal", action="store_true",
                        help="计算基因组数据的常规参数，如最小等位基因频率，缺失率等，并输出")
    parser.add_argument("--phe_cal", action="store_true",
                        help="计算标签数据的均值和标准差，并画出表型数值分布图，保存到当前目录")
    parser.add_argument("--importance", action="store_true",
                        help="输出模型训练后的特征重要性图和 CSV 文件")
    # 新增：保存检查点的参数
    parser.add_argument("--save_checkpoint", type=str, default="",
                        help="训练结束后保存模型的检查点文件路径（例如 checkpoint/model.m）")

    # 并行与 GPU 参数
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="并行计算使用的线程数")
    parser.add_argument("--gpu", action="store_true",
                        help="使用 GPU 进行训练，默认使用 CPU")
    
    # 自动化预测参数
    parser.add_argument("--auto", action="store_true",
                        help="启用自动化预测模式，自动比较多个模型并选择最佳")
    parser.add_argument("--auto_optimize", action="store_true", default=True,
                        help="在自动化模式下是否进行超参数优化")
    parser.add_argument("--auto_preprocess", action="store_true", default=True,
                        help="在自动化模式下是否进行数据预处理")

    args = parser.parse_args()

    # 解析 JSON 格式的参数（确保使用双引号，布尔值用小写）
    try:
        args.model_params = json.loads(args.model_params)
    except Exception:
        args.model_params = {}
    try:
        args.grid_model_params = json.loads(args.grid_model_params)
    except Exception:
        args.grid_model_params = {}
    try:
        args.ssa_model_params = json.loads(args.ssa_model_params)
    except Exception:
        args.ssa_model_params = {}
    if args.category_cols:
        try:
            args.category_cols = [int(x.strip()) for x in args.category_cols.split(",")]
        except Exception:
            args.category_cols = None

    return args



