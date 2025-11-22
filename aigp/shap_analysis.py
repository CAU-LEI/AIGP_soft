# aigp/shap_analysis.py
import shap
import matplotlib.pyplot as plt
import os
from .utils import Timer


def analyze_shap(model, X, feature_names=None, output=None,
                 shap_beeswarm=False, shap_feature_heatmap=False, shap_feature_waterfall=False,
                 top_features=None):
    """
    计算并可视化 SHAP 值。支持蜂群图、热图、瀑布图。

    参数：
      model: 已训练好的模型
      X: 用于计算 SHAP 值的特征数据（最好为原始数据或降维后数据）
      feature_names: 特征名称列表
      output: 保存图片的文件夹路径
      shap_beeswarm: 是否生成蜂群图
      shap_feature_heatmap: 是否生成特征热图
      shap_feature_waterfall: 是否生成瀑布图（仅针对单一样本）
      top_features: 显示前 N 个重要特征（用于部分图形）
    """
    print("开始SHAP分析...")
    
    # 限制样本数量以提高计算效率
    max_samples = 1000
    if len(X) > max_samples:
        print(f"数据样本数({len(X)})超过{max_samples}，将随机采样{max_samples}个样本进行SHAP分析")
        import numpy as np
        np.random.seed(42)
        sample_indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X.iloc[sample_indices] if hasattr(X, 'iloc') else X[sample_indices]
    else:
        X_sample = X
    
    # 选择解释器
    try:
        # 首先尝试TreeExplainer（适用于树模型）
        explainer = shap.TreeExplainer(model)
        print("使用TreeExplainer")
    except Exception as e:
        print(f"TreeExplainer失败: {e}")
        try:
            # 尝试LinearExplainer（适用于线性模型）
            explainer = shap.LinearExplainer(model, X_sample)
            print("使用LinearExplainer")
        except Exception as e2:
            print(f"LinearExplainer失败: {e2}")
            try:
                # 最后使用KernelExplainer（通用但较慢）
                print("使用KernelExplainer（可能较慢）")
                explainer = shap.KernelExplainer(model.predict, X_sample.iloc[:100] if hasattr(X_sample, 'iloc') else X_sample[:100])
            except Exception as e3:
                print(f"所有SHAP解释器都失败: {e3}")
                return None

    with Timer("计算 SHAP 值"):
        try:
            shap_values = explainer.shap_values(X_sample)
            print(f"SHAP值计算完成，形状: {np.array(shap_values).shape}")
        except Exception as e:
            print(f"SHAP值计算失败: {e}")
            return None

    # 若指定输出路径但不存在则创建
    if output and not os.path.exists(output):
        os.makedirs(output)

    # 蜂群图
    if shap_beeswarm:
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, max_display=top_features, show=False)
            if output:
                plt.savefig(os.path.join(output, "shap_beeswarm.png"), dpi=300, bbox_inches='tight')
                print("蜂群图已保存到:", os.path.join(output, "shap_beeswarm.png"))
            else:
                plt.show()
            plt.close()
        except Exception as e:
            print(f"蜂群图生成失败: {e}")

    # 特征热图
    if shap_feature_heatmap:
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, plot_type="heatmap", max_display=top_features, show=False)
            if output:
                plt.savefig(os.path.join(output, "shap_heatmap.png"), dpi=300, bbox_inches='tight')
                print("热图已保存到:", os.path.join(output, "shap_heatmap.png"))
            else:
                plt.show()
            plt.close()
        except Exception as e:
            print(f"热图生成失败: {e}")

    # 瀑布图（仅对单一样本有效）
    if shap_feature_waterfall:
        try:
            # 选择第一个样本
            idx = 0
            plt.figure(figsize=(10, 6))
            # 创建SHAP Explanation对象
            explanation = shap.Explanation(values=shap_values[idx], 
                                         base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                                         data=X_sample.iloc[idx] if hasattr(X_sample, 'iloc') else X_sample[idx],
                                         feature_names=feature_names)
            shap.waterfall_plot(explanation, max_display=top_features, show=False)
            if output:
                plt.savefig(os.path.join(output, "shap_waterfall.png"), dpi=300, bbox_inches='tight')
                print("瀑布图已保存到:", os.path.join(output, "shap_waterfall.png"))
            else:
                plt.show()
            plt.close()
        except Exception as e:
            print(f"瀑布图生成失败: {e}")

    return shap_values
